import datetime

elapsed_time1 = datetime.timedelta(hours=1)
elapsed_time2 = datetime.timedelta(hours=10)

print(
    str(elapsed_time1) < str(elapsed_time2)
)  # False, because "1:00:00" is lexicographically greater than "10:00:00"
