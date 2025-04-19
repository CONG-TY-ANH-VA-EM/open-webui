from typing import Dict, List, Optional, Union, Any, Callable
import json
import logging
import time
from dataclasses import dataclass, field

# Milvus imports with error handling
try:
    from pymilvus import MilvusClient as Client
    from pymilvus import FieldSchema, DataType, MilvusException
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    
from open_webui.retrieval.vector.main import VectorItem, SearchResult, GetResult
from open_webui.config import (
    MILVUS_URI,
    MILVUS_DB,
    MILVUS_TOKEN,
)
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

@dataclass
class MilvusConfig:
    """Configuration for Milvus client"""
    uri: str = MILVUS_URI
    db_name: str = MILVUS_DB
    token: Optional[str] = MILVUS_TOKEN
    collection_prefix: str = "open_webui"
    
    # Vector search parameters
    default_metric_type: str = "COSINE"
    normalize_distances: bool = True
    
    # Index parameters
    index_type: str = "HNSW"
    index_params: Dict[str, Any] = field(default_factory=lambda: {
        "M": 16,               # Number of bidirectional links
        "efConstruction": 100  # Higher value = better quality, slower build
    })
    
    # Performance parameters
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Query limits
    max_query_limit: int = 16383  # Milvus maximum records per request


class MilvusClient:
    """
    Advanced Milvus client for RAG operations with improved error handling,
    performance optimizations, and RAG-specific functionality.
    """
    
    def __init__(self, config: Optional[MilvusConfig] = None):
        """
        Initialize Milvus client with configuration
        
        Args:
            config: MilvusConfig object with client settings
        """
        if not MILVUS_AVAILABLE:
            log.error("pymilvus not available. Please install with 'pip install pymilvus'")
            raise ImportError("pymilvus is required but not installed")
            
        self.config = config or MilvusConfig()
        self._connected = False
        self._client = None
        self._initialize_client()
        
    def _initialize_client(self) -> None:
        """Initialize Milvus client with retry mechanism"""
        for attempt in range(self.config.max_retries):
            try:
                if self.config.token:
                    self._client = Client(
                        uri=self.config.uri, 
                        db_name=self.config.db_name, 
                        token=self.config.token
                    )
                else:
                    self._client = Client(
                        uri=self.config.uri, 
                        db_name=self.config.db_name
                    )
                    
                self._connected = True
                log.info(f"Connected to Milvus at {self.config.uri}, database: {self.config.db_name}")
                break
                
            except Exception as e:
                log.warning(f"Connection attempt {attempt+1}/{self.config.max_retries} failed: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    log.error(f"Failed to connect to Milvus after {self.config.max_retries} attempts")
                    raise ConnectionError(f"Could not connect to Milvus: {str(e)}")
    
    def _ensure_connected(self) -> None:
        """Ensure client is connected, reconnect if needed"""
        if not self._connected or self._client is None:
            self._initialize_client()
    
    def _normalize_collection_name(self, collection_name: str) -> str:
        """
        Normalize collection name to format required by Milvus
        
        Args:
            collection_name: Original collection name
            
        Returns:
            Normalized collection name with prefix
        """
        # Replace hyphens with underscores
        normalized = collection_name.replace("-", "_")
        # Add prefix if not already present
        prefixed_name = f"{self.config.collection_prefix}_{normalized}"
        return prefixed_name
    
    def _with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute an operation with retry logic
        
        Args:
            operation: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation
        """
        self._ensure_connected()
        
        for attempt in range(self.config.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                log.warning(f"Operation failed (attempt {attempt+1}/{self.config.max_retries}): {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    log.error(f"Operation failed after {self.config.max_retries} attempts")
                    raise
    
    def _result_to_get_result(self, result) -> GetResult:
        """
        Convert Milvus query result to GetResult format
        
        Args:
            result: Raw Milvus result
            
        Returns:
            Formatted GetResult object
        """
        ids = []
        documents = []
        metadatas = []

        for match in result:
            _ids = []
            _documents = []
            _metadatas = []
            for item in match:
                _ids.append(item.get("id"))
                _documents.append(item.get("data", {}).get("text"))
                _metadatas.append(item.get("metadata"))

            ids.append(_ids)
            documents.append(_documents)
            metadatas.append(_metadatas)

        return GetResult(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def _result_to_search_result(self, result) -> SearchResult:
        """
        Convert Milvus search result to SearchResult format
        
        Args:
            result: Raw Milvus result
            
        Returns:
            Formatted SearchResult object
        """
        ids = []
        distances = []
        documents = []
        metadatas = []

        for match in result:
            _ids = []
            _distances = []
            _documents = []
            _metadatas = []

            for item in match:
                _ids.append(item.get("id"))
                
                # Normalize distance based on metric type
                if self.config.normalize_distances:
                    # For cosine similarity, normalize from [-1, 1] to [0, 1]
                    # https://milvus.io/docs/de/metric.md
                    _dist = (item.get("distance") + 1.0) / 2.0
                else:
                    _dist = item.get("distance")
                    
                _distances.append(_dist)
                _documents.append(item.get("entity", {}).get("data", {}).get("text"))
                _metadatas.append(item.get("entity", {}).get("metadata"))

            ids.append(_ids)
            distances.append(_distances)
            documents.append(_documents)
            metadatas.append(_metadatas)

        return SearchResult(
            ids=ids,
            distances=distances,
            documents=documents,
            metadatas=metadatas,
        )

    def _create_collection(self, collection_name: str, dimension: int) -> None:
        """
        Create a new collection with RAG-optimized schema
        
        Args:
            collection_name: Name of collection (without prefix)
            dimension: Vector dimension
        """
        normalized_name = self._normalize_collection_name(collection_name)
        
        try:
            log.info(f"Creating new collection: {normalized_name} with dimension {dimension}")
            
            # Create schema optimized for RAG
            schema = self._client.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            
            # Primary key field
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=65535,
            )
            
            # Vector field
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=dimension,
                description="embedding vector",
            )
            
            # Text data field
            schema.add_field(
                field_name="data", 
                datatype=DataType.JSON, 
                description="text content and additional data"
            )
            
            # Metadata field for filtering
            schema.add_field(
                field_name="metadata", 
                datatype=DataType.JSON, 
                description="metadata for filtering and context"
            )

            # Prepare index parameters
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type=self.config.index_type,
                metric_type=self.config.default_metric_type,
                params=self.config.index_params,
            )

            # Create the collection with index
            self._client.create_collection(
                collection_name=normalized_name,
                schema=schema,
                index_params=index_params,
            )
            
            log.info(f"Successfully created collection {normalized_name} with {self.config.index_type} index")
            
        except Exception as e:
            log.error(f"Failed to create collection {normalized_name}: {str(e)}")
            raise

    def has_collection(self, collection_name: str) -> bool:
        """
        Check if collection exists
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            True if collection exists, False otherwise
        """
        normalized_name = self._normalize_collection_name(collection_name)
        
        try:
            return self._with_retry(self._client.has_collection, collection_name=normalized_name)
        except Exception as e:
            log.error(f"Error checking collection {normalized_name}: {str(e)}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if successful, False otherwise
        """
        normalized_name = self._normalize_collection_name(collection_name)
        
        try:
            if not self.has_collection(collection_name):
                log.warning(f"Collection {normalized_name} does not exist, nothing to delete")
                return False
                
            result = self._with_retry(self._client.drop_collection, collection_name=normalized_name)
            log.info(f"Deleted collection {normalized_name}")
            return True
        except Exception as e:
            log.error(f"Failed to delete collection {normalized_name}: {str(e)}")
            return False

    def search(
        self, 
        collection_name: str, 
        vectors: List[List[float]], 
        limit: int,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> Optional[SearchResult]:
        """
        Search for similar vectors in the collection
        
        Args:
            collection_name: Name of the collection to search
            vectors: Query vectors
            limit: Maximum number of results per query
            filter_expr: Optional filter expression
            output_fields: Optional fields to include in results
            
        Returns:
            SearchResult object with results or None if error
        """
        normalized_name = self._normalize_collection_name(collection_name)
        
        if not self.has_collection(collection_name):
            log.warning(f"Collection {normalized_name} does not exist")
            return None
            
        if not output_fields:
            output_fields = ["data", "metadata"]
            
        try:
            log.debug(f"Searching in collection {normalized_name} for {len(vectors)} vectors with limit {limit}")
            
            result = self._with_retry(
                self._client.search,
                collection_name=normalized_name,
                data=vectors,
                limit=limit,
                output_fields=output_fields,
                filter=filter_expr if filter_expr else None
            )
            
            return self._result_to_search_result(result)
            
        except Exception as e:
            log.error(f"Search failed in {normalized_name}: {str(e)}")
            return None

    def query(
        self, 
        collection_name: str, 
        filter: Dict[str, Any], 
        limit: Optional[int] = None
    ) -> Optional[GetResult]:
        """
        Query collection with filter
        
        Args:
            collection_name: Name of the collection
            filter: Filter dictionary
            limit: Maximum results to return (None for all)
            
        Returns:
            GetResult object with results or None if error
        """
        normalized_name = self._normalize_collection_name(collection_name)
        
        if not self.has_collection(collection_name):
            log.warning(f"Collection {normalized_name} does not exist")
            return None

        # Construct filter expression from dictionary
        filter_string = " && ".join(
            [f'metadata["{key}"] == {json.dumps(value)}' for key, value in filter.items()]
        )

        max_limit = self.config.max_query_limit
        all_results = []

        if limit is None:
            limit = float("inf")  # Use infinity as placeholder for no limit

        # Initialize pagination parameters
        offset = 0
        remaining = limit

        try:
            # Paginate through results
            while remaining > 0:
                log.debug(f"Querying {normalized_name} with filter, remaining: {remaining}")
                current_fetch = min(max_limit, remaining)

                results = self._with_retry(
                    self._client.query,
                    collection_name=normalized_name,
                    filter=filter_string,
                    output_fields=["*"],
                    limit=current_fetch,
                    offset=offset,
                )

                if not results:
                    break

                all_results.extend(results)
                results_count = len(results)
                remaining -= results_count
                offset += results_count

                # Break if returned fewer results than requested
                if results_count < current_fetch:
                    break

            log.debug(f"Query returned {len(all_results)} total results")
            return self._result_to_get_result([all_results])
            
        except Exception as e:
            log.error(f"Error querying collection {normalized_name}: {str(e)}")
            return None

    def get(self, collection_name: str) -> Optional[GetResult]:
        """
        Get all items in a collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            GetResult object with all items or None if error
        """
        normalized_name = self._normalize_collection_name(collection_name)
        
        if not self.has_collection(collection_name):
            log.warning(f"Collection {normalized_name} does not exist")
            return None
            
        try:
            log.debug(f"Getting all items from collection {normalized_name}")
            
            result = self._with_retry(
                self._client.query,
                collection_name=normalized_name,
                filter='id != ""',
            )
            
            return self._result_to_get_result([result])
            
        except Exception as e:
            log.error(f"Failed to get items from {normalized_name}: {str(e)}")
            return None

    def insert(self, collection_name: str, items: List[VectorItem]) -> bool:
        """
        Insert items into collection
        
        Args:
            collection_name: Name of the collection
            items: List of VectorItem objects to insert
            
        Returns:
            True if successful, False otherwise
        """
        if not items:
            log.warning("No items to insert")
            return True
            
        normalized_name = self._normalize_collection_name(collection_name)
        
        # Create collection if it doesn't exist
        try:
            if not self.has_collection(collection_name):
                self._create_collection(collection_name, dimension=len(items[0]["vector"]))
        except Exception as e:
            log.error(f"Failed to create collection {normalized_name}: {str(e)}")
            return False

        # Prepare data format for Milvus
        formatted_items = [
            {
                "id": item["id"],
                "vector": item["vector"],
                "data": {"text": item["text"]},
                "metadata": item["metadata"],
            }
            for item in items
        ]
        
        # Insert with batching for better performance
        batch_size = self.config.batch_size
        
        try:
            for i in range(0, len(formatted_items), batch_size):
                batch = formatted_items[i:i + batch_size]
                
                self._with_retry(
                    self._client.insert,
                    collection_name=normalized_name,
                    data=batch
                )
                
                log.debug(f"Inserted batch of {len(batch)} items into {normalized_name}")
                
            log.info(f"Successfully inserted {len(formatted_items)} items into {normalized_name}")
            return True
            
        except Exception as e:
            log.error(f"Failed to insert items into {normalized_name}: {str(e)}")
            return False

    def upsert(self, collection_name: str, items: List[VectorItem]) -> bool:
        """
        Upsert (insert or update) items in collection
        
        Args:
            collection_name: Name of the collection
            items: List of VectorItem objects to upsert
            
        Returns:
            True if successful, False otherwise
        """
        if not items:
            log.warning("No items to upsert")
            return True
            
        normalized_name = self._normalize_collection_name(collection_name)
        
        # Create collection if it doesn't exist
        try:
            if not self.has_collection(collection_name):
                self._create_collection(collection_name, dimension=len(items[0]["vector"]))
        except Exception as e:
            log.error(f"Failed to create collection {normalized_name}: {str(e)}")
            return False

        # Prepare data format for Milvus
        formatted_items = [
            {
                "id": item["id"],
                "vector": item["vector"],
                "data": {"text": item["text"]},
                "metadata": item["metadata"],
            }
            for item in items
        ]
        
        # Upsert with batching for better performance
        batch_size = self.config.batch_size
        
        try:
            for i in range(0, len(formatted_items), batch_size):
                batch = formatted_items[i:i + batch_size]
                
                self._with_retry(
                    self._client.upsert,
                    collection_name=normalized_name,
                    data=batch
                )
                
                log.debug(f"Upserted batch of {len(batch)} items into {normalized_name}")
                
            log.info(f"Successfully upserted {len(formatted_items)} items into {normalized_name}")
            return True
            
        except Exception as e:
            log.error(f"Failed to upsert items into {normalized_name}: {str(e)}")
            return False

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Delete items from collection
        
        Args:
            collection_name: Name of the collection
            ids: Optional list of IDs to delete
            filter: Optional filter dictionary
            
        Returns:
            True if successful, False otherwise
        """
        normalized_name = self._normalize_collection_name(collection_name)
        
        if not self.has_collection(collection_name):
            log.warning(f"Collection {normalized_name} does not exist")
            return False
            
        try:
            if ids:
                log.info(f"Deleting {len(ids)} items by ID from {normalized_name}")
                
                # Delete in batches for better performance
                batch_size = self.config.batch_size
                for i in range(0, len(ids), batch_size):
                    batch = ids[i:i + batch_size]
                    
                    self._with_retry(
                        self._client.delete,
                        collection_name=normalized_name,
                        ids=batch
                    )
                    
                return True
                
            elif filter:
                log.info(f"Deleting items by filter from {normalized_name}: {filter}")
                
                # Convert filter to expression
                filter_string = " && ".join(
                    [f'metadata["{key}"] == {json.dumps(value)}' for key, value in filter.items()]
                )
                
                self._with_retry(
                    self._client.delete,
                    collection_name=normalized_name,
                    filter=filter_string
                )
                
                return True
                
            else:
                log.warning("No IDs or filter provided for deletion")
                return False
                
        except Exception as e:
            log.error(f"Failed to delete items from {normalized_name}: {str(e)}")
            return False

    def reset(self) -> bool:
        """
        Reset database by deleting all collections with the configured prefix
        
        Returns:
            True if successful, False if errors occurred
        """
        self._ensure_connected()
        success = True
        
        try:
            collection_names = self._with_retry(self._client.list_collections)
            
            log.info(f"Resetting database. Found {len(collection_names)} collections")
            
            for collection_name in collection_names:
                if collection_name.startswith(self.config.collection_prefix):
                    try:
                        self._with_retry(self._client.drop_collection, collection_name=collection_name)
                        log.info(f"Deleted collection {collection_name}")
                    except Exception as e:
                        log.error(f"Failed to delete collection {collection_name}: {str(e)}")
                        success = False
                        
            return success
            
        except Exception as e:
            log.error(f"Failed to reset database: {str(e)}")
            return False

    def count(self, collection_name: str) -> int:
        """
        Get number of items in a collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Number of items or 0 if collection doesn't exist
        """
        normalized_name = self._normalize_collection_name(collection_name)
        
        if not self.has_collection(collection_name):
            return 0
            
        try:
            count = self._with_retry(
                self._client.query,
                collection_name=normalized_name,
                filter='id != ""',
                output_fields=["id"],
                limit=1
            )
            
            if count and hasattr(count, "get_row_count"):
                return count.get_row_count()
            else:
                # Fallback for older Milvus versions
                count_result = self._with_retry(
                    self._client.query,
                    collection_name=normalized_name,
                    filter='id != ""',
                    output_fields=["id"]
                )
                return len(count_result) if count_result else 0
                
        except Exception as e:
            log.error(f"Failed to get count for {normalized_name}: {str(e)}")
            return 0

    def list_collections(self) -> List[str]:
        """
        List all collections with the configured prefix
        
        Returns:
            List of collection names (without prefix)
        """
        self._ensure_connected()
        
        try:
            all_collections = self._with_retry(self._client.list_collections)
            
            # Filter collections by prefix and remove prefix
            prefix_len = len(self.config.collection_prefix) + 1  # +1 for underscore
            collections = [
                coll[prefix_len:] 
                for coll in all_collections 
                if coll.startswith(self.config.collection_prefix)
            ]
            
            return collections
            
        except Exception as e:
            log.error(f"Failed to list collections: {str(e)}")
            return []


# Initialize global client instance with error handling
try:
    VECTOR_DB_CLIENT = MilvusClient()
    log.info("Milvus client initialized successfully")
except Exception as e:
    log.error(f"Failed to initialize Milvus client: {str(e)}")
    
    # Create dummy client for graceful degradation
    class DummyMilvusClient:
        """Fallback client when Milvus is unavailable"""
        def __init__(self):
            log.warning("Using dummy Milvus client - vector operations will fail")
            
        def __getattr__(self, name):
            def method(*args, **kwargs):
                log.warning(f"Milvus not available: {name} operation skipped")
                if name == "has_collection":
                    return False
                elif name in ["search", "query", "get", "list_collections"]:
                    return None
                elif name == "count":
                    return 0
                else:
                    return False
            return method
    
    VECTOR_DB_CLIENT = DummyMilvusClient()
