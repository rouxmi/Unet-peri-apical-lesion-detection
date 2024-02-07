# import in a dataframes from a csv file
import pandas as pd

df = pd.read_csv("outputs/model_parameters.csv")
print(df.head())
print(df.shape)

# drop all the rows with any "error" in a test_dice column
df = df[df.test_dice != "error"]
print(df.head())
print(df.shape)

# convert the test_dice, validation_dice, validation_loss, test_loss columns to numeric
df["test_dice"] = pd.to_numeric(df["test_dice"])
df["validation_dice"] = pd.to_numeric(df["validation_dice"])
df["validation_loss"] = pd.to_numeric(df["validation_loss"])
df["test_loss"] = pd.to_numeric(df["test_loss"])

# sort the dataframe by test_dice column
df = df.sort_values(by=["test_dice"], ascending=False)
print(df.head())

# make some plots of the data for exploration
import matplotlib.pyplot as plt

# plot the test_dice column vs the test_loss column
plt.plot(df["test_dice"], df["test_loss"])
plt.xlabel("Dice")
plt.ylabel("Loss")
plt.title("Test Dice Score vs Test Loss")
plt.show()

# make a correlation matrix of the dataframe

# first, convert the act and norm columns to numeric by encoding them
act_values = df["act"].unique()
act_mapping = {act: i for i, act in enumerate(act_values)}
df["act"] = df["act"].map(act_mapping)

norm_values = df["norm"].unique()
norm_mapping = {norm: i for i, norm in enumerate(norm_values)}
df["norm"] = df["norm"].map(norm_mapping)

channels_values = df["channels"].unique()
channels_mapping = {channels: i for i, channels in enumerate(channels_values)}
df["channels"] = df["channels"].map(channels_mapping)

strides_values = df["strides"].unique()
strides_mapping = {strides: i for i, strides in enumerate(strides_values)}
df["strides"] = df["strides"].map(strides_mapping)
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.show()



# keep only the rows with a softmax activation function and INSTANCE_NVFUSER normalization
df = pd.read_csv("outputs/model_parameters.csv")
df = df[df.test_dice != "error"]
df = df[df.act == "SOFTMAX"]
df = df[df.norm == "INSTANCE_NVFUSER"]

# order the dataframe by test_loss 
df = df.sort_values(by=["test_loss"], ascending=True)

channels = df["channels"].tolist()
test_loss = df["test_loss"].tolist()
for i in range(len(channels)):
    channels[i] = len(str(channels[i]).split("(")[1].split(")")[0].split(", "))

plt.plot(channels, test_loss)
plt.xlabel("Channels")
plt.ylabel("Loss")
plt.title("Channels vs Test Loss")
plt.show()

# get a correlation matrix of the dataframe

