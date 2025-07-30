import re
import json
import argparse
import os
from typing import List, Dict
from persona import DISILLUSIONED_CANUCKS_FAN


SYSTEM_MESSAGES = [
    'image omitted',
    'media omitted',
    'messages and calls are end-to-end encrypted',
    'created this group',
    'you were added'
]


def redact_sensitive_info(text: str) -> str:
    """
    Redact any sensitive info from exported whatsapp chats.
    Regex courtesy of mr. sonnet 4.
    """

    phone_patterns = [
        r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',  # International format
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US format (XXX-XXX-XXXX)
        r'\b\d{10}\b',  # 10 digit numbers
        r'\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}',  # (XXX) XXX-XXXX format
    ]
    
    for pattern in phone_patterns:
        text = re.sub(pattern, '[PHONE_REDACTED]', text)
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, '[EMAIL_REDACTED]', text)

    return text

def parse_whatsapp_chat(file_path: str) -> List[Dict]:
    """
    Parse WhatsApp chat file and extract messages with timestamps and senders.
    Format: [YYYY-MM-DD, HH:MM:SS AM/PM] Sender Name: message
    """
    messages = []

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # clean up any invisible unicode characters
    content = re.sub(r'[\u200e\u200f\u202a-\u202e\u2060]', '', content)
    
    # Updated regex pattern to handle leading whitespace and special characters in sender names
    pattern = r'^\[(?P<datetime>\d{4}-\d{2}-\d{2}, \d{1,2}:\d{2}:\d{2} (?:AM|PM))\] (?P<sender>.*?): (?P<message>.*)$'

    matches = re.findall(pattern, content, re.MULTILINE)
    
    for timestamp_str, sender, message in matches:
        message = message.strip()
        sender = sender.strip()
        
        # skip empty messages or system messages
        if (not message) or (any(sm in message.lower() for sm in SYSTEM_MESSAGES)):
            continue

        # redact any sensitive info
        message = redact_sensitive_info(message)
        sender = redact_sensitive_info(sender)
        
        messages.append({
            'timestamp': timestamp_str,
            'sender': sender,
            'message': message
        })

    return messages

def create_training_data(messages: List[Dict], target_sender: str, context_window: int = 5) -> List[Dict]:
    """
    Create training data specifically for the target sender's responses.
    Only includes examples where the target sender is responding to the conversation.
    """
    training_data = []
    
    for i in range(len(messages)):
        current_message = messages[i]
        
        # only include if current message is from target sender
        if current_message['sender'] != target_sender:
            continue
            
        # get context messages (previous messages in conversation)
        context_messages = []
        context_start = max(0, i - context_window)
        
        for j in range(context_start, i):
            if messages[j]['sender'] != target_sender:
                context_messages.append(messages[j])
        
        # only create training example if we have some context
        if context_messages:
            # create conversation with system message
            conversation = [
                {"role": "system", "content": DISILLUSIONED_CANUCKS_FAN}
            ]
            
            # add context messages as user input
            for msg in context_messages:
                conversation.append({
                    "role": "user", 
                    "content": msg['message']
                })
            
            # add target sender's response
            conversation.append({
                "role": "assistant",
                "content": current_message['message']
            })
            
            training_data.append({
                "messages": conversation
            })
    
    return training_data

def save_to_jsonl(data: List[Dict], output_file: str):
    """
    Save training data to JSONL format.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def save_train_valid_split(data: List[Dict], output_dir: str):
    """
    Save training data split into train (90%) and validation (10%) JSONL files.
    """
    # Calculate split index
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]
    
    # Create output files
    train_file = os.path.join(output_dir, 'train.jsonl')
    valid_file = os.path.join(output_dir, 'valid.jsonl')
    
    # Save train data
    save_to_jsonl(train_data, train_file)
    print(f"Train data saved to: {train_file} ({len(train_data)} examples)")
    
    # Save validation data
    save_to_jsonl(valid_data, valid_file)
    print(f"Validation data saved to: {valid_file} ({len(valid_data)} examples)")
    
    return train_file, valid_file

def main():
    parser = argparse.ArgumentParser(description='Convert whatsapp chat to JSONL for LLM fine-tuning')
    parser.add_argument('input_file', nargs='?', default='data/input/whatsapp_chat.txt', 
                       help='Input WhatsApp chat file (default: data/input/whatsapp_chat.txt)')
    parser.add_argument('--output', '-o', default='data/',
                       help='Output directory (default: data/)')
    parser.add_argument('--target-sender',
                       help='Name of the sender to generate training data for.')
    parser.add_argument('--context-window', type=int, default=5,
                       help='Number of previous messages to include as context')
    parser.add_argument('--min-length', type=int, default=5,
                       help='Minimum message length to include')
    
    args = parser.parse_args()
    
    print(f"Parsing whatsapp chat from file: {args.input_file}.")
    
    messages = parse_whatsapp_chat(args.input_file)
    
    if not messages:
        print("Unable to parse any messages from input file.")
        return

    print(f"Found {len(messages)} messages.")
    
    # filter messages by min length
    messages = [msg for msg in messages if len(msg['message']) >= args.min_length]
    print(f"After filtering by minimum length ({args.min_length}): {len(messages)} messages.")
    
    # count messages from target sender
    target_messages = [msg for msg in messages if msg['sender'] == args.target_sender]
    print(f"Found {len(target_messages)} messages from target sender {args.target_sender}.")
    
    # create training data from target sender's responses
    training_data = create_training_data(messages, args.target_sender, args.context_window)
    
    print(f"Created {len(training_data)} training examples for target sender {args.target_sender}")
    
    # create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # save to train/valid split
    train_file, valid_file = save_train_valid_split(training_data, args.output)
    
    # print sample data
    if training_data:
        print("Sample training example:")
        print(json.dumps(training_data[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()