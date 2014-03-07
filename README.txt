This dataset contains tweets from the following 4 organitions: 

Training set:
Apple (95 positive, 202 negative, 324 neutral) 
Google (125 positive, 29 negative, 117 neutral)
Microsoft (56 positive, 79 negative, 400 neutral)
Twitter (31 positive, 33 negative, 348 neutral)
Total: 1839

Testing set:
Apple (49 positive, 92 negative, 164 neutral) 
Google (14 positive, 16 negative, 60 neutral)
Microsoft (30 positive, 41 negative, 202 neutral)
Twitter (17 positive, 18 negative, 175 neutral)
Total: 928

The data is in json format, which contains all available information provided by Twitter.
For details about the defination of each field, please refer to
https://dev.twitter.com/docs/platform-objects/tweets
If you need to get more information (e.g., the social links between users), you could use the Twitter API:
https://dev.twitter.com/docs/api/1.1

A simple java program is provided to demonstrate how to read the json data, get specific data field, and generate word index.

A new file with several new tweets (in the same format with the test file) will be used to test your program during presentation. 
Please notice that the new tweets may contain words that are never shown in the training and testing dataset.

This dataset contains original data crowled from Twitter. 
Due to privacy issues, please do not public this dataset to anyone or for any use outside the class. Thank you.