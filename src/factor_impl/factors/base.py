from abc import ABC, abstractmethod
import pandas as pd


class BaseFactor(ABC):
    """量化因子基类，所有因子需继承此类并实现 compute 方法。"""

    name: str = ""
    category: str = ""
    description: str = ""

    @abstractmethod
    def compute(self, **kwargs) -> pd.DataFrame:
        """计算因子值。

        Args:
            **kwargs: 因子计算所需的数据，具体参数由子类定义。

        Returns:
            pd.DataFrame: 因子值，index 为日期，columns 为股票代码。
        """
        pass

    def get_metadata(self) -> dict:
        """返回因子元信息。"""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', category='{self.category}')"
