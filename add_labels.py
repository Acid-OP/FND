import pandas as pd

# Define the name of your input file and the desired output file
input_filename = './Dataset/Fake.csv'
output_filename = 'Fake_with_labels.csv'

try:
    # Read your original CSV file into a pandas DataFrame.
    # If your file is tab-separated (as we discussed earlier), use this line:
    # df = pd.read_csv(input_filename, delimiter='\t')
    #
    # If you have re-downloaded the file and it is now a standard comma-separated file, use this line:
    df = pd.read_csv(input_filename)

    # Add a new column named 'label' and set its value to 'FAKE' for every row
    df['label'] = 'FAKE'

    # Save the modified DataFrame to a new CSV file.
    # index=False prevents pandas from writing the DataFrame index as a column.
    df.to_csv(output_filename, index=False)

    print(f"Successfully created the file '{output_filename}' with the new 'label' column.")
    print("\nHere's a preview of the first 5 rows:")
    print(df.head())

except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found.")
    print("Please make sure the script is in the same directory as your Fake.csv file, or provide the full path to the file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")