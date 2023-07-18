import pandas as pd

# Create an empty DataFrame to store the results
result_df = pd.DataFrame(columns=['Dataset', 'totle', 'Negative Sample', 'Positive Sample', 'Sample proportion'])

File = {
    "./Dataset_M/": ["Doench.csv", "CRISPOR.csv", "SITE-Seq.csv", "GUIDE-Seq_Tasi.csv",
                      "GUIDE-Seq_Kleinstiver.csv", "GUIDE-Seq_Listgarten.csv",
                      "K562.csv", "Hek293t.csv"],
    "./Dataset_IM/": ["CIRCLE_seq.csv", "Listgarten.csv"]
}

for filepath in File:
    for filename in File[filepath]:
        # Read CSV file
        df = pd.read_csv(filepath + filename)

        # Calculate label counts
        label_counts = df['label'].value_counts()

        # Append the results to the DataFrame
        result_df = result_df.append({
            'Dataset': filename,
            'totle': len(df),
            'Negative Sample': label_counts[0],
            'Positive Sample': label_counts[1],
            'Sample proportion': f"{int(label_counts[0] / label_counts[1])}:1"
        }, ignore_index=True)

        # Output individual results
        print('Label 0 number:', label_counts[0], '，percentage：', label_counts[0] / len(df))
        print('Label 1 number：', label_counts[1], '，percentage：', label_counts[1] / len(df))
        print(filename, 'Sum', len(df), 'proportions：', int(label_counts[0] / label_counts[1]), ":1\n\n")

# Export the DataFrame to Excel
result_df.to_excel('NP.xlsx', index=False)
