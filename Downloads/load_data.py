import pandas as pd

def read_data():
    # Reading the csv file into a dataframe
    output_filename = f"results/output_load_data.txt"
    outfile = open(output_filename, 'w', encoding='utf-8')
    hotels = pd.read_csv('data/hotels_dataset.csv', encoding='ISO-8859-1')
    print("***********************Reading the Data***********************\n", file = outfile)
    print(hotels.head(), file = outfile)
    
    print("\n***********************Looking at the Number of Rows and Columns***********************\n", file = outfile)
    
    print("\nNo of Rows: {}".format(hotels.shape[0]), file = outfile)
    print("No of Columns: {}".format(hotels.shape[1]), file = outfile)
    
    #The column names have some spaces so we need to remove the spaces
    hotels.columns = hotels.columns.str.strip()
    
    return hotels

