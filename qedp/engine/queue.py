"""Event queue implementation with priority ordering."""
import heapq
from typing import List
from qedp.events.base import Event


class EventQueue:
    """
    Priority queue for events.
    
    Events are ordered by:
    1. Timestamp (earlier first)
    2. Priority (higher first)
    """
    
    def __init__(self, max_size: int = 100000):
        self._queue: List[Event] = []
        self.max_size = max_size
        
    def push(self, event: Event):
        """Add an event to the queue."""
        if len(self._queue) >= self.max_size:
            raise RuntimeError(f"Event queue overflow: {len(self._queue)} >= {self.max_size}")
        heapq.heappush(self._queue, event)
        
    def push_batch(self, events: List[Event]):
        """Add multiple events (micro-batch)."""
        for event in events:
            self.push(event)
    
    def pop(self) -> Event | None:
        """Remove and return the highest priority event."""
        if self._queue:
            return heapq.heappop(self._queue)
        return None
    
    def peek(self) -> Event | None:
        """View the highest priority event without removing it."""
        if self._queue:
            return self._queue[0]
        return None
    
    def __len__(self) -> int:
        return len(self._queue)
    
    def is_empty(self) -> bool:
        return len(self._queue) == 0
    
    def clear(self):
        """Remove all events from the queue."""
        self._queue.clear()
