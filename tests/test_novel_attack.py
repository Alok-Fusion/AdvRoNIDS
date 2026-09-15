import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.app import load_runtime_resources, generate_novel_attack, generate_report, ReportRequest

def test_novel_attack_and_report():
    load_runtime_resources()
    
    # 1. Test generate_novel_attack()
    data = generate_novel_attack()
    assert "attack_name" in data, "attack_name missing"
    assert "base_category" in data, "base_category missing"
    assert "forensics" in data, "forensics missing"
    assert "executive_summary" in data["forensics"], "executive_summary missing in forensics"
    print(f"PASS: Generated Novel Attack: {data['attack_name']} ({data['base_category']})")
    print(f"Executive Summary: {data['forensics']['executive_summary']}")
    
    # 2. Test generate_report with attack_data
    req = ReportRequest(report_type="attack", attack_data=data)
    report_res = generate_report(req)
    assert report_res.media_type == "application/pdf"
    assert len(report_res.body) > 1000, "PDF content is too small"
    print(f"PASS: Attack Threat Report PDF generated successfully ({len(report_res.body)} bytes)!")

if __name__ == "__main__":
    test_novel_attack_and_report()
    print("\nAll novel attack and attack-specific report tests PASSED successfully!")
