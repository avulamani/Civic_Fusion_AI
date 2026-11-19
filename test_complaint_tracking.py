"""Automated verification suite for CivicFusion AI Complaint Tracking.

Tests all 7 scenarios mandated in the task specification:
1. Valid ticket ID: CF-1043
2. Invalid / nonexistent ticket ID
3. Empty / whitespace ticket ID
4. Complaint with missing optional fields
5. Complaints with long descriptions and remarks
6. Complaints with special characters and HTML-like tags in user content
7. Complaint updated by a municipal officer (status, remarks, assignment, last_updated)
"""

from datetime import date
import json
from pathlib import Path
import sys
import unittest

from complaint_view import (
    format_complaint_card_html,
    get_status_theme,
    prepare_complaint_data,
)
from storage import (
    load_all_complaints,
    save_all_complaints,
    update_ticket_status,
)


class TestComplaintTracking(unittest.TestCase):
    def setUp(self):
        # Save a backup of current complaints so test updates don't corrupt real data
        self.original_complaints = load_all_complaints()
        # Ensure test ticket CF-1043 exists for reproducible testing
        if not any(c.get("id") == "CF-1043" for c in self.original_complaints):
            test_ticket = {
                "id": "CF-1043",
                "reported_date": "2026-11-17",
                "category_display": "Garbage / Waste",
                "problem_title": "Garbage / Waste",
                "description": "parking gate waste",
                "location_name": "parking gate",
                "city": "City Zone",
                "latitude": 31.25176,
                "longitude": 75.70381,
                "status": "Open",
                "department": "Sanitation & Waste Management Department",
                "priority": "P2",
                "severity": "High",
                "citizen_reports_count": 1,
            }
            save_all_complaints([test_ticket] + self.original_complaints)

    def tearDown(self):
        # Restore original complaints
        save_all_complaints(self.original_complaints)

    def test_scenario_1_valid_ticket_cf_1043(self):
        """Scenario 1: Valid ticket ID (CF-1043)."""
        complaints = load_all_complaints()
        ticket = next((c for c in complaints if c.get("id") == "CF-1043"), None)
        self.assertIsNotNone(ticket, "CF-1043 must exist in storage")

        # Citizen view HTML
        html_citizen = format_complaint_card_html(ticket, is_officer=False)
        self.assertIn("Ticket CF-1043", html_citizen)
        self.assertIn("Garbage / Waste", html_citizen)
        self.assertIn("parking gate", html_citizen)
        self.assertIn("Sanitation &amp; Waste Management Department", html_citizen)
        self.assertIn("Open", html_citizen)

        # Must NOT contain markdown indented code block triggers (>= 4 leading spaces on any line)
        for i, line in enumerate(html_citizen.split("\n")):
            self.assertFalse(
                line.startswith("    "),
                f"Line {i} has 4+ leading spaces: {line[:30]}..."
            )

        # Officer view HTML
        html_officer = format_complaint_card_html(ticket, is_officer=True)
        self.assertIn("Ticket CF-1043", html_officer)
        self.assertIn("Open Incident Coordinates in Google Maps", html_officer)

    def test_scenario_2_invalid_nonexistent_ticket(self):
        """Scenario 2: Invalid or nonexistent ticket ID."""
        complaints = load_all_complaints()
        invalid_id = "CF-9999-NONEXISTENT"
        found = next((c for c in complaints if str(c.get("id", "")).upper() == invalid_id), None)
        self.assertIsNone(found, "Nonexistent ticket must return None")

    def test_scenario_3_empty_ticket_id(self):
        """Scenario 3: Empty ticket ID input."""
        empty_queries = ["", "   ", "\t", "\n"]
        complaints = load_all_complaints()
        for q in empty_queries:
            clean_q = q.strip().upper()
            self.assertEqual(clean_q, "")
            # Empty query must never match any ticket
            found = next((c for c in complaints if str(c.get("id", "")).strip().upper() == clean_q), None)
            self.assertIsNone(found)

    def test_scenario_4_missing_optional_fields(self):
        """Scenario 4: Complaint with missing optional fields."""
        sparse_complaint = {
            "id": "CF-SPARSE",
            # No description, no location, no coordinates, no category, no remarks, no officer, no photo
        }
        data = prepare_complaint_data(sparse_complaint)
        self.assertEqual(data["id"], "CF-SPARSE")
        self.assertEqual(data["description"], "No description provided.")
        self.assertEqual(data["location_name"], "Unspecified Location")
        self.assertIsNone(data["latitude"])
        self.assertIsNone(data["longitude"])
        self.assertIsNone(data["assigned_officer"])
        self.assertIsNone(data["officer_remarks"])
        self.assertEqual(data["status"], "Open")

        html_out = format_complaint_card_html(sparse_complaint, is_officer=False)
        self.assertIn("CF-SPARSE", html_out)
        self.assertIn("Unspecified Location", html_out)
        self.assertIn("No description provided.", html_out)

    def test_scenario_5_long_description_and_remarks(self):
        """Scenario 5: Complaints with long descriptions and remarks."""
        long_desc = "Hazardous road condition: " + ("deep pothole causing traffic obstruction and vehicle damage. " * 20)
        long_remarks = "Officer inspection update: " + ("Field team inspected and placed warning cones. " * 20)
        complaint = {
            "id": "CF-LONG",
            "description": long_desc,
            "officer_remarks": long_remarks,
            "location_name": "Major Highway Sector 4",
            "status": "In Progress",
            "reported_date": "2026-11-09",
            "last_updated": "2026-11-10",
        }
        html_out = format_complaint_card_html(complaint, is_officer=False)
        self.assertIn("CF-LONG", html_out)
        self.assertIn("word-break:break-word", html_out)
        self.assertIn(long_desc[:50], html_out)
        self.assertIn(long_remarks[:50], html_out)

    def test_scenario_6_special_characters_and_xss_safety(self):
        """Scenario 6: Special characters or HTML-like text in user-submitted content."""
        malicious_input = '<script>alert("XSS Attack!")</script><table border="1"><tr><td>Injected</td></tr></table>'
        complaint = {
            "id": "CF-SECURITY",
            "description": f"Problem with {malicious_input} & \"quotes\" and 'apostrophes'",
            "location_name": "Intersection <Main & 5th>",
            "officer_remarks": f"Follow-up: {malicious_input}",
            "status": "Pending Review",
        }
        html_out = format_complaint_card_html(complaint, is_officer=False)

        # Raw executable script or raw table tags must NOT be present unescaped
        self.assertNotIn('<script>alert("XSS Attack!")</script>', html_out)
        self.assertNotIn('<table border="1">', html_out)

        # Escaped equivalents MUST be present
        self.assertIn('&lt;script&gt;alert(&quot;XSS Attack!&quot;)&lt;/script&gt;', html_out)
        self.assertIn('&lt;table border=&quot;1&quot;&gt;', html_out)
        self.assertIn('Intersection &lt;Main &amp; 5th&gt;', html_out)

    def test_scenario_7_officer_status_update_and_remarks(self):
        """Scenario 7: Complaint whose status has been updated by a municipal officer."""
        target_id = "CF-1043"
        new_status = "In Progress"
        officer_name = "Insp. R. Sharma"
        remarks_text = "Sanitation crew dispatched to parking gate. Heavy bins cleared; sanitization underway."

        # Perform officer update
        updated_list = update_ticket_status(
            ticket_id=target_id,
            new_status=new_status,
            remarks=remarks_text,
            assigned_officer=officer_name,
        )

        # Verify in returned list
        updated_ticket = next((c for c in updated_list if c["id"] == target_id), None)
        self.assertIsNotNone(updated_ticket)
        self.assertEqual(updated_ticket["status"], new_status)
        self.assertEqual(updated_ticket["assigned_officer"], officer_name)
        self.assertEqual(updated_ticket["officer_remarks"], remarks_text)
        self.assertEqual(updated_ticket["last_updated"], date.today().isoformat())

        # Verify in disk re-read
        persisted_list = load_all_complaints()
        persisted_ticket = next((c for c in persisted_list if c["id"] == target_id), None)
        self.assertEqual(persisted_ticket["status"], new_status)
        self.assertEqual(persisted_ticket["assigned_officer"], officer_name)
        self.assertEqual(persisted_ticket["officer_remarks"], remarks_text)

        # Verify formatted HTML displays updated information
        html_citizen = format_complaint_card_html(persisted_ticket, is_officer=False)
        self.assertIn("In Progress", html_citizen)
        self.assertIn(officer_name, html_citizen)
        self.assertIn(remarks_text, html_citizen)
        self.assertIn(date.today().isoformat(), html_citizen)


if __name__ == "__main__":
    unittest.main()

