"""Main event-driven engine."""
from typing import Dict, List, Callable
import logging

from qedp.events.base import (
    Event, MarketEvent, SignalEvent, OrderEvent, CancelEvent,
    FillEvent, AccountEvent, ClockEvent, ControlEvent
)
from qedp.engine.queue import EventQueue
from qedp.engine.clock import Clock

logger = logging.getLogger(__name__)


class Engine:
    """
    Core event-driven engine.
    
    Main loop: pop event -> route -> pump -> maybe_tick_clock
    """
    
    def __init__(self, clock: Clock, event_queue: EventQueue):
        self.clock = clock
        self.event_queue = event_queue
        self.running = False
        
        # Event handlers registry
        self._handlers: Dict[type, List[Callable]] = {
            MarketEvent: [],
            SignalEvent: [],
            OrderEvent: [],
            CancelEvent: [],
            FillEvent: [],
            AccountEvent: [],
            ClockEvent: [],
            ControlEvent: []
        }
        
    def register_handler(self, event_type: type, handler: Callable):
        """Register a handler for a specific event type."""
        if event_type not in self._handlers:
            raise ValueError(f"Unknown event type: {event_type}")
        self._handlers[event_type].append(handler)
        
    def route(self, event: Event):
        """Route event to registered handlers."""
        event_type = type(event)
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in handler for {event_type.__name__}: {e}", exc_info=True)
        else:
            logger.warning(f"No handlers registered for {event_type.__name__}")
    
    def pump(self):
        """
        Process all events currently in queue until stable.
        
        This allows handlers to generate new events that are processed
        in the same time step.
        """
        max_iterations = 1000  # Prevent infinite loops
        iteration = 0
        
        while not self.event_queue.is_empty() and iteration < max_iterations:
            event = self.event_queue.pop()
            if event:
                self.route(event)
            iteration += 1
            
        if iteration >= max_iterations:
            logger.warning("Pump reached max iterations - possible event loop")
    
    def maybe_tick_clock(self):
        """Advance clock and inject ClockEvent if phase boundary crossed."""
        clock_event = self.clock.tick()
        if clock_event:
            self.event_queue.push(clock_event)
    
    def run(self):
        """Main event loop."""
        self.running = True
        logger.info("Engine starting...")
        
        while self.running:
            # Check for events in queue
            if not self.event_queue.is_empty():
                event = self.event_queue.pop()
                if event:
                    # Handle control events
                    if isinstance(event, ControlEvent):
                        self._handle_control(event)
                    else:
                        self.route(event)
                    
                    # Process any derived events
                    self.pump()
            else:
                # No events - advance clock
                self.maybe_tick_clock()
                
                # If still no events and clock is done, stop
                if self.event_queue.is_empty():
                    clock_event = self.clock.tick()
                    if clock_event is None:
                        # Clock has finished
                        logger.info("Clock finished - stopping engine")
                        break
                    else:
                        self.event_queue.push(clock_event)
        
        logger.info("Engine stopped")
    
    def _handle_control(self, event: ControlEvent):
        """Handle control events."""
        if event.cmd == "stop":
            logger.info("Received stop command")
            self.running = False
        elif event.cmd == "pause":
            logger.info("Received pause command")
            # TODO: Implement pause logic
        elif event.cmd == "resume":
            logger.info("Received resume command")
            # TODO: Implement resume logic
        elif event.cmd == "reload":
            logger.info(f"Received reload command: {event.args}")
            # TODO: Implement config reload
        else:
            logger.warning(f"Unknown control command: {event.cmd}")
    
    def stop(self):
        """Stop the engine."""
        self.running = False
