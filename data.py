# Scrape Data using built-in package such as: Selenium

# Format into csv files

# Possibly combine csv files into a master csv

# Generate more specific csv's from master for specific metrics

# Is there a data structure we can use to store all csv's?

import csv

class raw_data:

    # PROPERTIES
    # nickname -> A name for the dataset for reference purposes
    # start_year -> Starting year of data set
    # end_year -> Ending year of data set
    # file_name -> csv file that stores the data
    # n_rows -> number of rows in the data

    def __init__(self, nickname, start_year, end_year, file_name):
        self.nickname = nickname
        self.start_year = start_year
        self.end_year = end_year
        self.file_name = file_name
        self.rows = []
        with open(file_name, 'r') as csvfile:
            # create reader
            csvreader = csv.reader(csvfile)

            # fill fields array with top row
            self.fields = next(csvreader)

            # fill rows
            for row in csvreader:
                self.rows.append(row)

            self.n_rows = csvreader.line_num

    def print_table(self):
        print("Now printing ", self.nickname)
        headers = ""
        for field in self.fields:
            headers += field + '\t'
        print(headers)
        for row in self.rows:
            row_values = ""
            for field in row:
                row_values += field + '\t'
            print(row_values)







