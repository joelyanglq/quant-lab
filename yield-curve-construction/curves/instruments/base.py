"""
金融工具基类

定义所有金融工具的通用接口和基础功能
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Union
import numpy as np


class Instrument(ABC):
    """
    金融工具基类
    
    所有具体的金融工具都应该继承此类并实现抽象方法
    """
    
    def __init__(self, issue_date: Union[date, str], maturity_date: Union[date, str]):
        """
        初始化金融工具
        
        Args:
            issue_date: 发行日期
            maturity_date: 到期日期
        """
        self.issue_date = self._to_date(issue_date)
        self.maturity_date = self._to_date(maturity_date)
        
        if self.maturity_date <= self.issue_date:
            raise ValueError("到期日期必须晚于发行日期")
    
    def _to_date(self, date_input: Union[date, str]) -> date:
        """将输入转换为date对象"""
        if isinstance(date_input, str):
            return datetime.strptime(date_input, '%Y-%m-%d').date()
        return date_input
    
    @property
    def time_to_maturity(self) -> float:
        """剩余期限（年）"""
        today = date.today()
        days_to_maturity = (self.maturity_date - today).days
        return max(0, days_to_maturity / 365.25)
    
    @abstractmethod
    def price(self, yield_rate: float) -> float:
        """
        根据收益率计算价格
        
        Args:
            yield_rate: 年化收益率
            
        Returns:
            价格
        """
        pass
    
    @abstractmethod
    def yield_to_price(self, price: float) -> float:
        """
        根据价格计算收益率
        
        Args:
            price: 价格
            
        Returns:
            年化收益率
        """
        pass
    
    @abstractmethod
    def cashflows(self) -> tuple[list[date], list[float]]:
        """
        获取现金流
        
        Returns:
            tuple: (现金流日期列表, 现金流金额列表)
        """
        pass
    
    def present_value(self, discount_curve: callable, t: float = 0.0) -> float:
        """
        使用折现曲线计算现值
        
        Args:
            discount_curve: 折现函数，输入时间t，输出折现因子
            t: 当前时间（年）
            
        Returns:
            现值
        """
        dates, amounts = self.cashflows()
        today = date.today()
        
        pv = 0.0
        for cash_date, amount in zip(dates, amounts):
            time_to_cashflow = max(0, (cash_date - today).days / 365.25 - t)
            discount_factor = discount_curve(time_to_cashflow)
            pv += amount * discount_factor
        
        return pv
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.issue_date} -> {self.maturity_date})"
    
    def __repr__(self):
        return self.__str__()
