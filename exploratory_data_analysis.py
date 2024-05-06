import matplotlib.pyplot as plt
from wordcloud import WordCloud

def conduct_eda(df):
    
    """
    Conducts exploratory data analysis on the provided DataFrame.

    Args:
    df (DataFrame): The DataFrame containing the data.

    Returns:
    None
    """
    
    output_filename = f"results/output_exploratory_data_analysis.txt"
    outfile = open(output_filename, 'w', encoding='utf-8')
    print("***********************Looking at the Target Variable Distribution***********************\n", file = outfile)
    print(df['HotelRating'].value_counts(), file = outfile)
    
    plt.figure(figsize=(8, 6))
    df['HotelRating'].value_counts().plot(kind='bar')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/target_variable_distribution.png')
    
    print("\n***********************Number of five star hotels in each city***********************\n", file = outfile)
    print(df[['HotelRating', 'cityName']][df['HotelRating'] == 'FiveStar'].value_counts(), file = outfile)
    
    print("\n***********************Number of one star hotels in each city***********************\n", file = outfile)
    print(df[['HotelRating', 'cityName']][df['HotelRating'] == 'OneStar'].value_counts(), file = outfile)
    
    df['HotelFacilities'].fillna('', inplace=True)

    text = ' '.join(df['HotelFacilities'])

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('results/feature_variable_wordcloud.png')
    plt.show()
    
    print("\n***********************Looking at the Target Variable Distribution after Normalization***********************\n", file = outfile)
    print(df['HotelRating'].value_counts(normalize=True), file = outfile)