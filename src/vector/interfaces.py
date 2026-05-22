from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Literal

VectorMode = Literal["hybrid", "dense", "sparse"]


class BaseVectorDB(ABC):
    @abstractmethod
    def create_collection(self, *args, **kwargs):
        pass

    @abstractmethod
    def upsert(self, points: List[Any]):
        pass

    @abstractmethod
    def search(self,
               dense_vec: Optional[List] = None,
               sparse_vec: Optional[Any] = None,
               limit: int = 5,
               filter_dict: Dict = None):
        pass


class BaseEmbeder(ABC):
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        pass

    @abstractmethod
    def compute_vectors(self,
                        text: str,
                        mode: VectorMode = "hybrid",
                        dense_vec_name: str = "dense",
                        sparse_vec_name: str = "sparse",
                        is_query: bool = False) -> Dict[str, Any]:
        pass
