import telethon
from telethon.sync import TelegramClient,ERRORS
from telethon.errors import SessionExpiredError, FloodWaitError
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import time
import nltk

# Your Telegram API ID and bot token
api_id = 123456
api_hash = 'YOUR_API_HASH'
bot_token = '7067106225:AAHU9ZzgFry-OGo-nJ0tKb9pwn5TPd79MIM'

# Create a Telegram client
client = TelegramClient('my_bot', api_id, api_hash)

# Login to Telegram
client.login(bot_token=bot_token)

# Get the target group ID
target_group_id = 'YOUR_TARGET_GROUP_ID'

# Get group information
group = client.get_chat(target_group_id)

# Get group members
members = client.get_participants(target_group_id)

# Initialize counters
total_members = 0
online_members = 0
fake_members_count = 0

# Initialize data structures
member_data = []
message_data = []

# Analyze members
for member in members:
    total_members += 1

    # Check if member is online
    if member.is_online:
        online_members += 1

    # Extract member information
    member_info = {
        'user_id': member.id,
        'first_name': member.first_name,
        'last_name': member.last_name,
        'username': member.username,
        'is_bot': member.is_bot,
        'last_joined': member.last_joined,
        'status': member.status
    }
    member_data.append(member_info)

# Get messages from the group
messages = client.get_messages(target_group_id, limit=1000)

# Analyze messages
for message in messages:
    message_info = {
        'sender_id': message.sender_id,
        'message_text': message.message,
        'date': message.date
    }
    message_data.append(message_info)

# Prepare data for fake member detection
X = []  # Features for each member
y = []  # Labels (0: real, 1: fake)

# Extract features from member data
for member_info in member_data:
    features = []

    # Check if member has a name
    if member_info['first_name'] or member_info['last_name']:
        features.append(1)
    else:
        features.append(0)

    # Check if member has a username
    if member_info['username']:
        features.append(1)
    else:
        features.append(0)

    # Check if member is a bot
    if member_info['is_bot']:
        features.append(1)
    else:
        features.append(0)

    # Check how long ago the member joined the group
    time_since_joined = (time.time() - member_info['last_joined'].timestamp()) / 86400
    if time_since_joined < 7:
        features.append(1)  # Joined recently
    else:
        features.append(0)

    X.append(features)

# Extract labels from message data
for message_info in message_data:
    if message_info['sender_id'] == client.get_me().id:
        y.append(0)  # Real member (bot)
    else:
        y.append(1)  # Potential fake member

# Train a naive Bayes classifier to detect fake members
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict labels for all members
predicted_labels = classifier.predict(X)

# Update fake member count
for i, predicted_label in enumerate(predicted_labels):
    if predicted_label == 1:
        fake_members_count += 1

# Calculate percentage of fake members
fake_members_percentage = (fake_members_count / total_members) * 100

# Prepare data for activity analysis
member_activity = []

# Analyze member activity
for member_info, predicted_label in zip(member_data, predicted_labels):
    member_activity_data = {
        'user_id': member_info['user_id'],
        'is_fake': predicted_label,
        'first_name': member_info['first_name'],
        'last_name': member_info['last_name'],
        'username': member_info['username'],
        'message_count': 0,
        'reply_count': 0,
        'post_count': 0
    }

    # Count messages for the member
    for message_info in message_data:
        if message_info['sender_id'] == member_info['user_id']:
            member_activity_data['message_count'] += 1
            if message_info['reply_to']:
                member_activity_data['reply_count'] += 1
            else:
                member_activity_data['post_count'] += 1

    member_activity.append(member_activity_data)

# Convert member activity data to pandas DataFrame
member_activity_df = pd.DataFrame(member_activity)

# Calculate average activity metrics
average_message_count = member_activity_df['message_count'].mean()
average_reply_count = member_activity_df['reply_count'].mean()
average_post_count = member_activity_df['post_count'].mean()

# Analyze member activity based on fake member status
fake_member_activity_df = member_activity_df[member_activity_df['is_fake'] == 1]
real_member_activity_df = member_activity_df[member_activity_df['is_fake'] == 0]

average_fake_message_count = fake_member_activity_df['message_count'].mean()
average_fake_reply_count = fake_member_activity_df['reply_count'].mean()
average_fake_post_count = fake_member_activity_df['post_count'].mean()

average_real_message_count = real_member_activity_df['message_count'].mean()
average_real_reply_count = real_member_activity_df['reply_count'].mean()
average_real_post_count = real_member_activity_df['post_count'].mean()

# Generate reports

# 1. Group overview report
group_overview_report = f"""
**Group Overview Report**

Group Name: {group.title}
Group ID: {group.id}
Total Members: {total_members}
Online Members: {online_members}
Fake Members: {fake_members_count} ({fake_members_percentage:.1f}%)

**Activity Metrics**

Average Message Count: {average_message_count:.1f}
Average Reply Count: {average_reply_count:.1f}
Average Post Count: {average_post_count:.1f}

**Fake Member Activity**

Average Fake Message Count: {average_fake_message_count:.1f}
Average Fake Reply Count: {average_fake_reply_count:.1f}
Average Fake Post Count: {average_fake_post_count:.1f}

**Real Member Activity**

Average Real Message Count: {average_real_message_count:.1f}
Average Real Reply Count: {average_real_reply_count:.1f}
Average Real Post Count: {average_real_post_count:.1f}
"""

print(group_overview_report)

# 2. Member activity report
member_activity_report = member_activity_df.to_string()
print(member_activity_report)

# 3. Fake member list
fake_members_list = member_activity_df[member_activity_df['is_fake'] == 1][['first_name', 'last_name', 'username']]
# 3. Fake member list (continued)
print(fake_members_list)

# 4. Visualizations

# Create pie chart for member distribution (real vs. fake)
labels = ['Real Members', 'Fake Members']
sizes = [total_members - fake_members_count, fake_members_count]
colors = ['lightblue', 'lightcoral']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title(f'Member Distribution: {group.title}')
plt.show()

# Create bar chart for average activity metrics (real vs. fake)
activity_metrics = ['Message Count', 'Reply Count', 'Post Count']
real_member_averages = [average_real_message_count, average_real_reply_count, average_real_post_count]
fake_member_averages = [average_fake_message_count, average_fake_reply_count, average_fake_post_count]

x = range(len(activity_metrics))
width = 0.35

plt.bar(x - width/2, real_member_averages, width=width, label='Real Members', color='lightblue')
plt.bar(x + width/2, fake_member_averages, width=width, label='Fake Members', color='lightcoral')
plt.xlabel('Activity Metric')
plt.ylabel('Average Count')
plt.title('Average Activity Metrics (Real vs. Fake)')
plt.xticks(x, activity_metrics)
plt.legend()
plt.show()

# Generate and save reports to files (optional)
with open('group_overview_report.txt', 'w') as f:
    f.write(group_overview_report)

member_activity_df.to_csv('member_activity_report.csv', index=False)

fake_members_list.to_csv('fake_members_list.csv', index=False)
