# check_healthy_header.py

FILEPATH = "../data/oral_only_results_metaphlan4.txt"  # مسیر فایل Healthy را اینجا بده

# فقط 5 خط اول را بخوان
with open(FILEPATH, 'r') as f:
    for i in range(5):
        line = f.readline().strip()
        print(f"Line {i+1}: {line}")
