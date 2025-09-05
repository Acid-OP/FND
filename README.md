1.Made styles agent class which takes news as input and gives a score according to the reliability of the news based on its stylistic features, Used qwen2.5-0.5b-instruct. 


2. Used ROC method to find the threshold. If the score is below threshold the news is real else it is fake.

3. Used CUDA 12.6 and the corresponding pytorch version which can be downloaded by a single command(can ask from chatgpt)

4. Accuracy:  around 53% which basically means it is random guessing.

So, take this as a boiler plate code and make other agents for other features.