from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status, Request
import logging

from open_webui.models.knowledge import (
    Knowledges,
    KnowledgeForm,
    KnowledgeResponse,
    KnowledgeUserResponse,
)
from open_webui.models.files import Files, FileModel
from open_webui.retrieval.vector.connector import VECTOR_DB_CLIENT
from open_webui.routers.retrieval import (
    process_file,
    ProcessFileForm,
    process_files_batch,
    BatchProcessFilesForm,
)
from open_webui.storage.provider import Storage

from open_webui.constants import ERROR_MESSAGES
from open_webui.utils.auth import get_verified_user
from open_webui.utils.access_control import has_access, has_permission


from open_webui.env import SRC_LOG_LEVELS
from open_webui.models.models import Models, ModelForm


log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter()

# ----- FastAPI Dependencies -----

async def get_knowledge_base(knowledge_id: str):
    """Get knowledge base by ID or raise 404"""
    knowledge = Knowledges.get_knowledge_by_id(id=knowledge_id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
    return knowledge

async def validate_knowledge_access(knowledge: KnowledgeResponse, user: Any, access_type: str = "read"):
    """Validate user access to a knowledge base"""
    if knowledge.user_id != user.id and not has_access(user.id, access_type, knowledge.access_control) and user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )
    return knowledge

async def get_knowledge_with_access(id: str, user: Any, access_type: str = "read"):
    """Get knowledge base and check access in a single dependency"""
    knowledge = await get_knowledge_base(id)
    return await validate_knowledge_access(knowledge, user, access_type)

# ----- Helper Functions -----

def get_files_with_cleanup(knowledge: KnowledgeResponse):
    """Get files for a knowledge base and clean up missing files"""
    files = []
    if not knowledge.data:
        return files, []
    
    file_ids = knowledge.data.get("file_ids", [])
    if not file_ids:
        return files, []
        
    files = Files.get_file_metadatas_by_ids(file_ids)
    
    # Check if all files exist
    if len(files) != len(file_ids):
        missing_files = list(
            set(file_ids) - set([file.id for file in files])
        )
        if missing_files:
            data = knowledge.data.copy()
            clean_file_ids = [id for id in file_ids if id not in missing_files]
            data["file_ids"] = clean_file_ids
            Knowledges.update_knowledge_data_by_id(
                id=knowledge.id, data=data
            )
            return files, clean_file_ids
    
    return files, file_ids

# ----- Response Models -----

class KnowledgeFilesResponse(KnowledgeResponse):
    files: list[FileModel]
    warnings: Optional[Dict[str, Any]] = None

class KnowledgeFileIdForm(BaseModel):
    file_id: str

# ----- Knowledge Base Endpoints -----

@router.get("/", response_model=list[KnowledgeUserResponse])
async def get_knowledge(user=Depends(get_verified_user)):
    """Get all knowledge bases accessible to the user with read access"""
    knowledge_bases = []

    if user.role == "admin":
        knowledge_bases = Knowledges.get_knowledge_bases()
    else:
        knowledge_bases = Knowledges.get_knowledge_bases_by_user_id(user.id, "read")

    # Get files for each knowledge base
    knowledge_with_files = []
    for knowledge_base in knowledge_bases:
        files, _ = get_files_with_cleanup(knowledge_base)
        knowledge_with_files.append(
            KnowledgeUserResponse(
                **knowledge_base.model_dump(),
                files=files,
            )
        )

    return knowledge_with_files


@router.get("/list", response_model=list[KnowledgeUserResponse])
async def get_knowledge_list(user=Depends(get_verified_user)):
    """Get all knowledge bases accessible to the user with write access"""
    knowledge_bases = []

    if user.role == "admin":
        knowledge_bases = Knowledges.get_knowledge_bases()
    else:
        knowledge_bases = Knowledges.get_knowledge_bases_by_user_id(user.id, "write")

    # Get files for each knowledge base
    knowledge_with_files = []
    for knowledge_base in knowledge_bases:
        files, _ = get_files_with_cleanup(knowledge_base)
        knowledge_with_files.append(
            KnowledgeUserResponse(
                **knowledge_base.model_dump(),
                files=files,
            )
        )
    return knowledge_with_files


@router.post("/create", response_model=Optional[KnowledgeResponse])
async def create_new_knowledge(
    request: Request, form_data: KnowledgeForm, user=Depends(get_verified_user)
):
    """Create a new knowledge base"""
    if user.role != "admin" and not has_permission(
        user.id, "workspace.knowledge", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    knowledge = Knowledges.insert_new_knowledge(user.id, form_data)

    if knowledge:
        return knowledge
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.FILE_EXISTS,
        )


@router.post("/{id}/reindex", response_model=Dict[str, Any])
async def reindex_knowledge_base(
    request: Request,
    id: str,
    user=Depends(get_verified_user)
):
    """Reindex a specific knowledge base"""
    knowledge = await get_knowledge_with_access(id, user, "write")
    
    log.info(f"Starting reindexing for knowledge base {id} ({knowledge.name})")
    
    # Get files for this knowledge base
    files = []
    if knowledge.data:
        file_ids = knowledge.data.get("file_ids", [])
        files = Files.get_files_by_ids(file_ids)
    
    results = {
        "success": True,
        "total_files": len(files),
        "successful_files": 0,
        "failed_files": 0,
        "errors": []
    }
    
    # Delete the collection if it exists
    try:
        if VECTOR_DB_CLIENT.has_collection(collection_name=id):
            VECTOR_DB_CLIENT.delete_collection(collection_name=id)
            log.info(f"Deleted collection {id}")
    except Exception as e:
        log.error(f"Error deleting collection {id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting vector DB collection: {str(e)}",
        )
    
    # Process each file
    for file in files:
        try:
            process_file(
                request,
                ProcessFileForm(file_id=file.id, collection_name=id),
                user=user,
            )
            results["successful_files"] += 1
            log.info(f"Successfully processed file {file.id}")
        except Exception as e:
            results["failed_files"] += 1
            results["errors"].append({"file_id": file.id, "error": str(e)})
            log.error(f"Error processing file {file.id}: {str(e)}")
    
    return results


@router.post("/reindex", response_model=Dict[str, Any])
async def reindex_all_knowledge_bases(request: Request, user=Depends(get_verified_user)):
    """Reindex all knowledge bases (admin only)"""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    knowledge_bases = Knowledges.get_knowledge_bases()
    log.info(f"Starting reindexing for {len(knowledge_bases)} knowledge bases")
    
    results = {
        "total_knowledge_bases": len(knowledge_bases),
        "successful": 0,
        "failed": 0,
        "details": {}
    }

    for knowledge_base in knowledge_bases:
        kb_result = {
            "name": knowledge_base.name,
            "success": True,
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "errors": []
        }
        
        try:
            # Get files for this knowledge base
            files = []
            if knowledge_base.data:
                file_ids = knowledge_base.data.get("file_ids", [])
                files = Files.get_files_by_ids(file_ids)
            
            kb_result["total_files"] = len(files)
            
            # Delete the collection if it exists
            try:
                if VECTOR_DB_CLIENT.has_collection(collection_name=knowledge_base.id):
                    VECTOR_DB_CLIENT.delete_collection(collection_name=knowledge_base.id)
            except Exception as e:
                log.error(f"Error deleting collection {knowledge_base.id}: {str(e)}")
                raise Exception(f"Error preparing vector database: {str(e)}")
            
            # Process each file
            for file in files:
                try:
                    process_file(
                        request,
                        ProcessFileForm(file_id=file.id, collection_name=knowledge_base.id),
                        user=user,
                    )
                    kb_result["successful_files"] += 1
                except Exception as e:
                    kb_result["failed_files"] += 1
                    kb_result["errors"].append({"file_id": file.id, "error": str(e)})
                    log.error(f"Error processing file {file.id}: {str(e)}")
            
            results["successful"] += 1
        except Exception as e:
            kb_result["success"] = False
            kb_result["error"] = str(e)
            results["failed"] += 1
            log.error(f"Error processing knowledge base {knowledge_base.id}: {str(e)}")
        
        results["details"][knowledge_base.id] = kb_result

    return results


@router.get("/{id}", response_model=Optional[KnowledgeFilesResponse])
async def get_knowledge_by_id(id: str, user=Depends(get_verified_user)):
    """Get knowledge base by ID"""
    knowledge = await get_knowledge_with_access(id, user, "read")
    files = Files.get_files_by_ids(knowledge.data.get("file_ids", []) if knowledge.data else [])
    
    return KnowledgeFilesResponse(
        **knowledge.model_dump(),
        files=files,
    )


@router.post("/{id}/update", response_model=Optional[KnowledgeFilesResponse])
async def update_knowledge_by_id(
    id: str,
    form_data: KnowledgeForm,
    user=Depends(get_verified_user),
):
    """Update knowledge base"""
    knowledge = await get_knowledge_with_access(id, user, "write")
    
    updated_knowledge = Knowledges.update_knowledge_by_id(id=id, form_data=form_data)
    if not updated_knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ID_TAKEN,
        )
    
    files = Files.get_files_by_ids(updated_knowledge.data.get("file_ids", []) if updated_knowledge.data else [])
    
    return KnowledgeFilesResponse(
        **updated_knowledge.model_dump(),
        files=files,
    )


@router.post("/{id}/file/add", response_model=Optional[KnowledgeFilesResponse])
def add_file_to_knowledge_by_id(
    request: Request,
    id: str,
    form_data: KnowledgeFileIdForm,
    user=Depends(get_verified_user),
):
    """Add a file to knowledge base"""
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # Validate access
    if knowledge.user_id != user.id and not has_access(user.id, "write", knowledge.access_control) and user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    # Validate file exists
    file = Files.get_file_by_id(form_data.file_id)
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
    
    # Validate file is processed
    if not file.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.FILE_NOT_PROCESSED,
        )

    # Check if file already in knowledge base
    data = knowledge.data or {}
    file_ids = data.get("file_ids", [])
    if form_data.file_id in file_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("File already in knowledge base"),
        )

    # Add content to vector database
    try:
        process_file(
            request,
            ProcessFileForm(file_id=form_data.file_id, collection_name=id),
            user=user,
        )
    except Exception as e:
        log.error(f"Error processing file {form_data.file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Update knowledge base with new file ID
    file_ids.append(form_data.file_id)
    data["file_ids"] = file_ids
    
    updated_knowledge = Knowledges.update_knowledge_data_by_id(id=id, data=data)
    if not updated_knowledge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Failed to update knowledge base"),
        )
    
    files = Files.get_files_by_ids(file_ids)
    return KnowledgeFilesResponse(
        **updated_knowledge.model_dump(),
        files=files,
    )


@router.post("/{id}/file/update", response_model=Optional[KnowledgeFilesResponse])
def update_file_from_knowledge_by_id(
    request: Request,
    id: str,
    form_data: KnowledgeFileIdForm,
    user=Depends(get_verified_user),
):
    """Update a file in knowledge base"""
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # Validate access
    if knowledge.user_id != user.id and not has_access(user.id, "write", knowledge.access_control) and user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    # Validate file exists
    file = Files.get_file_by_id(form_data.file_id)
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # Remove content from vector database
    try:
        VECTOR_DB_CLIENT.delete(
            collection_name=knowledge.id, filter={"file_id": form_data.file_id}
        )
        log.info(f"Deleted existing vectors for file {form_data.file_id}")
    except Exception as e:
        log.warning(f"Error deleting vectors for file {form_data.file_id}: {str(e)}")

    # Add content to vector database
    try:
        process_file(
            request,
            ProcessFileForm(file_id=form_data.file_id, collection_name=id),
            user=user,
        )
        log.info(f"Successfully reprocessed file {form_data.file_id}")
    except Exception as e:
        log.error(f"Error processing file {form_data.file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Get files for knowledge base
    data = knowledge.data or {}
    file_ids = data.get("file_ids", [])
    files = Files.get_files_by_ids(file_ids)

    return KnowledgeFilesResponse(
        **knowledge.model_dump(),
        files=files,
    )


@router.post("/{id}/file/remove", response_model=Optional[KnowledgeFilesResponse])
def remove_file_from_knowledge_by_id(
    id: str,
    form_data: KnowledgeFileIdForm,
    user=Depends(get_verified_user),
):
    """Remove a file from knowledge base"""
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # Validate access
    if knowledge.user_id != user.id and not has_access(user.id, "write", knowledge.access_control) and user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    # Validate file exists
    file = Files.get_file_by_id(form_data.file_id)
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # Remove content from vector database
    try:
        VECTOR_DB_CLIENT.delete(
            collection_name=knowledge.id, filter={"file_id": form_data.file_id}
        )
        log.info(f"Deleted vectors for file {form_data.file_id}")
    except Exception as e:
        log.debug(f"Error deleting vectors for file {form_data.file_id}: {str(e)}")
        log.debug("This was most likely caused by bypassing embedding processing")

    # Remove file's collection if it exists
    try:
        file_collection = f"file-{form_data.file_id}"
        if VECTOR_DB_CLIENT.has_collection(collection_name=file_collection):
            VECTOR_DB_CLIENT.delete_collection(collection_name=file_collection)
            log.info(f"Deleted file collection {file_collection}")
    except Exception as e:
        log.debug(f"Error deleting file collection {file_collection}: {str(e)}")
        log.debug("This was most likely caused by bypassing embedding processing")

    # Delete file from database
    Files.delete_file_by_id(form_data.file_id)

    # Update knowledge base
    data = knowledge.data or {}
    file_ids = data.get("file_ids", [])
    
    if form_data.file_id in file_ids:
        file_ids.remove(form_data.file_id)
        data["file_ids"] = file_ids
        
        updated_knowledge = Knowledges.update_knowledge_data_by_id(id=id, data=data)
        if not updated_knowledge:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Failed to update knowledge base"),
            )
        
        files = Files.get_files_by_ids(file_ids)
        return KnowledgeFilesResponse(
            **updated_knowledge.model_dump(),
            files=files,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("File not in knowledge base"),
        )


@router.delete("/{id}/delete", response_model=bool)
async def delete_knowledge_by_id(id: str, user=Depends(get_verified_user)):
    """Delete knowledge base"""
    knowledge = await get_knowledge_with_access(id, user, "write")
    log.info(f"Deleting knowledge base: {id} (name: {knowledge.name})")

    # Update models that reference this knowledge base
    models = Models.get_all_models()
    log.info(f"Found {len(models)} models to check for knowledge base {id}")

    for model in models:
        if model.meta and hasattr(model.meta, "knowledge"):
            knowledge_list = model.meta.knowledge or []
            # Filter out the deleted knowledge base
            updated_knowledge = [k for k in knowledge_list if k.get("id") != id]
            
            # If the knowledge list changed, update the model
            if len(updated_knowledge) != len(knowledge_list):
                log.info(f"Updating model {model.id} to remove knowledge base {id}")
                model.meta.knowledge = updated_knowledge
                # Create a ModelForm for the update
                model_form = ModelForm(
                    id=model.id,
                    name=model.name,
                    base_model_id=model.base_model_id,
                    meta=model.meta,
                    params=model.params,
                    access_control=model.access_control,
                    is_active=model.is_active,
                )
                Models.update_model_by_id(model.id, model_form)

    # Clean up vector DB
    try:
        VECTOR_DB_CLIENT.delete_collection(collection_name=id)
        log.info(f"Deleted vector collection for knowledge base {id}")
    except Exception as e:
        log.warning(f"Error deleting collection {id}: {str(e)}")
    
    result = Knowledges.delete_knowledge_by_id(id=id)
    return result


@router.post("/{id}/reset", response_model=Optional[KnowledgeResponse])
async def reset_knowledge_by_id(id: str, user=Depends(get_verified_user)):
    """Reset knowledge base (clear all files)"""
    knowledge = await get_knowledge_with_access(id, user, "write")

    try:
        if VECTOR_DB_CLIENT.has_collection(collection_name=id):
            VECTOR_DB_CLIENT.delete_collection(collection_name=id)
            log.info(f"Deleted vector collection for knowledge base {id}")
    except Exception as e:
        log.warning(f"Error deleting collection {id}: {str(e)}")

    updated_knowledge = Knowledges.update_knowledge_data_by_id(id=id, data={"file_ids": []})
    return updated_knowledge


@router.post("/{id}/files/batch/add", response_model=Optional[KnowledgeFilesResponse])
def add_files_to_knowledge_batch(
    request: Request,
    id: str,
    form_data: list[KnowledgeFileIdForm],
    user=Depends(get_verified_user),
):
    """Add multiple files to a knowledge base"""
    knowledge = Knowledges.get_knowledge_by_id(id=id)
    if not knowledge:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # Validate access
    if knowledge.user_id != user.id and not has_access(user.id, "write", knowledge.access_control) and user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    # Get files
    log.info(f"Batch processing {len(form_data)} files for knowledge base {id}")
    files = []
    for form in form_data:
        file = Files.get_file_by_id(form.file_id)
        if not file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {form.file_id} not found",
            )
        files.append(file)

    # Process files
    try:
        result = process_files_batch(
            request=request,
            form_data=BatchProcessFilesForm(files=files, collection_name=id),
            user=user,
        )
    except Exception as e:
        log.error(f"Error in batch processing: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # Add successful files to knowledge base
    data = knowledge.data or {}
    existing_file_ids = data.get("file_ids", [])

    # Only add files that were successfully processed
    successful_file_ids = [r.file_id for r in result.results if r.status == "completed"]
    for file_id in successful_file_ids:
        if file_id not in existing_file_ids:
            existing_file_ids.append(file_id)

    data["file_ids"] = existing_file_ids
    knowledge = Knowledges.update_knowledge_data_by_id(id=id, data=data)

    # Get updated files
    files = Files.get_files_by_ids(existing_file_ids)

    # If there were any errors, include them in the response
    if result.errors:
        error_details = [f"{err.file_id}: {err.error}" for err in result.errors]
        return KnowledgeFilesResponse(
            **knowledge.model_dump(),
            files=files,
            warnings={
                "message": "Some files failed to process",
                "errors": error_details,
            },
        )

    return KnowledgeFilesResponse(
        **knowledge.model_dump(), files=files
    )
