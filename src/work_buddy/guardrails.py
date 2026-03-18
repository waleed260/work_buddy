"""
Guardrails for Remote Work Buddy AI Agent.
Ensures suggestions respect time zones, privacy, and work-life balance.
"""

from datetime import datetime
from typing import Optional


class Guardrails:
    """Guardrails to ensure safe and balanced recommendations."""
    
    def __init__(self, timezone: str = "Asia/Karachi", work_end_hour: int = 20):
        """
        Initialize guardrails.
        
        Args:
            timezone: User's timezone (default: PKT - Asia/Karachi)
            work_end_hour: Hour when work should end (default: 20 = 8 PM)
        """
        self.timezone = timezone
        self.work_end_hour = work_end_hour
    
    def validate_time_slot(self, hour: int) -> tuple[bool, Optional[str]]:
        """
        Validate if a time slot is within acceptable work hours.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if hour >= self.work_end_hour:
            return False, f"Task scheduled after {self.work_end_hour}:00 ({self.timezone}). Work-life balance: no tasks after 8 PM."
        if hour < 6:
            return False, "Task scheduled before 6:00 AM. Consider respecting rest hours."
        return True, None
    
    def validate_meeting_duration(self, duration_minutes: int) -> tuple[bool, Optional[str]]:
        """Validate meeting duration isn't too long."""
        if duration_minutes > 120:
            return False, "Meeting exceeds 2 hours. Consider breaking into shorter sessions."
        return True, None
    
    def validate_break_frequency(self, hours_since_last_break: float) -> tuple[bool, Optional[str]]:
        """Validate if a break is needed."""
        if hours_since_last_break >= 2:
            return False, "Break needed! More than 2 hours since last break."
        return True, None
    
    def check_privacy(self, content: str) -> tuple[bool, list[str]]:
        """Check for potential privacy concerns in content."""
        concerns = []
        
        sensitive_patterns = [
            ("password", "Contains potential password reference"),
            ("api_key", "Contains API key reference"),
            ("secret", "Contains secret reference"),
            ("token", "Contains token reference"),
            ("credit_card", "Contains credit card reference"),
            ("ssn", "Contains SSN reference"),
        ]
        
        content_lower = content.lower()
        for pattern, message in sensitive_patterns:
            if pattern in content_lower:
                concerns.append(message)
        
        return len(concerns) == 0, concerns
    
    def get_current_time_pkt(self) -> datetime:
        """Get current time in PKT timezone."""
        # For simplicity, using UTC+5 for PKT
        from datetime import timezone, timedelta
        pkt = timezone(timedelta(hours=5))
        return datetime.now(pkt)
    
    def is_within_work_hours(self) -> bool:
        """Check if current time is within work hours."""
        current = self.get_current_time_pkt()
        return 6 <= current.hour < self.work_end_hour
    
    def suggest_optimal_focus_time(self) -> str:
        """Suggest optimal focus session time."""
        current = self.get_current_time_pkt()
        
        # Morning focus block (9-11 AM)
        if current.hour < 11:
            return "09:00-11:00 (Morning focus block)"
        # Afternoon focus block (2-4 PM)
        elif current.hour < 16:
            return "14:00-16:00 (Afternoon focus block)"
        # Next day suggestion
        else:
            return "09:00-11:00 tomorrow (Morning focus block)"
    
    def format_time_for_pkt(self, hour: int, minute: int = 0) -> str:
        """Format time for PKT timezone."""
        return f"{hour:02d}:{minute:02d} PKT"
