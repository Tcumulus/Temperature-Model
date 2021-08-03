import csv

def append_list_as_row(list_of_elem):
    with open(f"/Users/Maarten/Documents/Projects/Temperature-Model/Data/Final_data/{int(list_of_elem[0])}_{int(list_of_elem[1])}.txt", 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(list_of_elem)

with open("/Users/Maarten/Documents/Projects/Temperature-Model/Data/Filtered_data/Combined_filtered_data.txt") as file:
    print("TEST")
    reader = csv.reader(file, delimiter=',')
    [append_list_as_row(row) for row in reader]
        