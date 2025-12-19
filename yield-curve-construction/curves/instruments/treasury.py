"""
美国国债工具类

实现不同类型的美国国债：
- TreasuryBill: 短期国债（T-Bills，零息债券）
- TreasuryNote: 中期国债（T-Notes，付息债券）
- TreasuryBond: 长期国债（T-Bonds，付息债券）
"""

from datetime import date, timedelta
from typing import Union, List, Tuple
import numpy as np
from scipy.optimize import newton
from .base import Instrument


class TreasuryBill(Instrument):
    """
    短期国债（T-Bills）
    
    零息债券，到期时支付面值
    """
    
    def __init__(self, issue_date: Union[date, str], maturity_date: Union[date, str], face_value: float = 100.0):
        """
        初始化短期国债
        
        Args:
            issue_date: 发行日期
            maturity_date: 到期日期
            face_value: 面值（默认100）
        """
        super().__init__(issue_date, maturity_date)
        self.face_value = face_value
    
    def price(self, yield_rate: float) -> float:
        """
        根据收益率计算价格（贴现公式）
        
        P = F / (1 + y * t/360)
        
        Args:
            yield_rate: 年化收益率
            
        Returns:
            价格
        """
        days_to_maturity = (self.maturity_date - self.issue_date).days
        t = days_to_maturity / 360.0  # 使用360天基准
        return self.face_value / (1 + yield_rate * t)
    
    def yield_to_price(self, price: float) -> float:
        """
        根据价格计算收益率
        
        y = (F/P - 1) * (360/t)
        
        Args:
            price: 价格
            
        Returns:
            年化收益率
        """
        days_to_maturity = (self.maturity_date - self.issue_date).days
        t = days_to_maturity / 360.0
        return (self.face_value / price - 1) * (360.0 / days_to_maturity)
    
    def cashflows(self) -> Tuple[List[date], List[float]]:
        """获取现金流（到期支付面值）"""
        return [self.maturity_date], [self.face_value]


class TreasuryNote(Instrument):
    """
    中期国债（T-Notes）
    
    付息债券，定期支付利息，到期支付本金
    """
    
    def __init__(self, issue_date: Union[date, str], maturity_date: Union[date, str], 
                 coupon_rate: float, face_value: float = 100.0, frequency: int = 2):
        """
        初始化中期国债
        
        Args:
            issue_date: 发行日期
            maturity_date: 到期日期
            coupon_rate: 年票息率
            face_value: 面值（默认100）
            frequency: 付息频率（每年付息次数，默认2次）
        """
        super().__init__(issue_date, maturity_date)
        self.coupon_rate = coupon_rate
        self.face_value = face_value
        self.frequency = frequency
        self.coupon_payment = face_value * coupon_rate / frequency
    
    def _generate_coupon_dates(self) -> List[date]:
        """生成付息日期"""
        dates = []
        current_date = self.issue_date
        
        # 向后生成付息日期直到到期日
        while current_date < self.maturity_date:
            # 添加6个月（或12/frequency个月）
            months_to_add = 12 // self.frequency
            current_date = self._add_months(current_date, months_to_add)
            if current_date <= self.maturity_date:
                dates.append(current_date)
        
        # 如果最后一个付息日期不是到期日，添加到期日
        if dates and dates[-1] != self.maturity_date:
            dates.append(self.maturity_date)
        elif not dates:
            dates.append(self.maturity_date)
        
        return dates
    
    def _add_months(self, date_obj: date, months: int) -> date:
        """给日期添加月份"""
        month = date_obj.month - 1 + months
        year = date_obj.year + month // 12
        month = month % 12 + 1
        day = min(date_obj.day, self._days_in_month(year, month))
        return date(year, month, day)
    
    def _days_in_month(self, year: int, month: int) -> int:
        """获取指定月份的天数"""
        if month == 2:
            if year % 400 == 0 or (year % 100 != 0 and year % 4 == 0):
                return 29
            return 28
        elif month in [4, 6, 9, 11]:
            return 30
        else:
            return 31
    
    def price(self, yield_rate: float) -> float:
        """
        根据收益率计算价格（债券定价公式）
        
        P = Σ(C/(1+y/f)^t) + F/(1+y/f)^T
        
        Args:
            yield_rate: 年化收益率
            
        Returns:
            价格
        """
        coupon_dates = self._generate_coupon_dates()
        today = date.today()
        
        # 计算每个现金流的现值
        price = 0.0
        y_per_period = yield_rate / self.frequency
        
        for i, coupon_date in enumerate(coupon_dates):
            days = (coupon_date - today).days
            t = days / 365.25  # 年
            periods = t * self.frequency
            
            if coupon_date == self.maturity_date:
                # 最后一次支付本金+利息
                cashflow = self.coupon_payment + self.face_value
            else:
                # 仅支付利息
                cashflow = self.coupon_payment
            
            price += cashflow / ((1 + y_per_period) ** periods)
        
        return price
    
    def yield_to_price(self, price: float) -> float:
        """
        根据价格计算收益率（使用牛顿法求解）
        
        Args:
            price: 价格
            
        Returns:
            年化收益率
        """
        def price_diff(y):
            return self.price(y) - price
        
        # 初始猜测：使用票息率
        initial_guess = self.coupon_rate
        
        try:
            yield_rate = newton(price_diff, initial_guess, maxiter=100, tol=1e-10)
            return yield_rate
        except:
            # 如果牛顿法失败，使用简单搜索
            for y in np.linspace(0.001, 0.5, 1000):
                if abs(self.price(y) - price) < 1e-6:
                    return y
            raise ValueError("无法计算收益率")
    
    def cashflows(self) -> Tuple[List[date], List[float]]:
        """获取现金流"""
        coupon_dates = self._generate_coupon_dates()
        amounts = []
        
        for coupon_date in coupon_dates:
            if coupon_date == self.maturity_date:
                # 最后一次支付本金+利息
                amounts.append(self.coupon_payment + self.face_value)
            else:
                # 仅支付利息
                amounts.append(self.coupon_payment)
        
        return coupon_dates, amounts


class TreasuryBond(TreasuryNote):
    """
    长期国债（T-Bonds）
    
    与中期国债类似，但期限更长
    """
    
    def __init__(self, issue_date: Union[date, str], maturity_date: Union[date, str], 
                 coupon_rate: float, face_value: float = 100.0, frequency: int = 2):
        """
        初始化长期国债
        
        Args:
            issue_date: 发行日期
            maturity_date: 到期日期
            coupon_rate: 年票息率
            face_value: 面值（默认100）
            frequency: 付息频率（默认2次）
        """
        super().__init__(issue_date, maturity_date, coupon_rate, face_value, frequency)
