import os
import subprocess

# Convert markdown to PDF
subprocess.run(["pandoc", "../report.md", "-o", "../results/report.pdf"])
print("PDF report generated!")