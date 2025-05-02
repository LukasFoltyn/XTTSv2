import threading
import queue
import time
import sys
import os

# For character-by-character input
import tty
import termios

# Shared queue for communication between threads
word_queue = queue.Queue()
# List to keep track of words that haven't been processed yet
pending_words = []
# Flag to signal threads to terminate
exit_flag = threading.Event()

def getchar():
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    tty.setraw(fd)
    ch = sys.stdin.read(1)
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
  return ch

def input_thread():
    """Thread that reads user input character by character and adds words to the queue."""
    print("Input thread started. Type words (space sends them to the queue).")
    print("Use backspace to delete characters. Type '!exit' to exit program.")
    
    current_word = ""
    sys.stdout.write("> ")
    sys.stdout.flush()
    
    while not exit_flag.is_set():
        try:
            char = getchar()
            
            # Handle backspace (different codes for different systems)
            if char in ('\b', '\x7f', '\x08'):
                if current_word:
                    current_word = current_word[:-1]
                    # Erase the last character on the screen
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
                continue
                
            # Echo the character back to the screen
            sys.stdout.write(char)
            sys.stdout.flush()
            
            # Check for exit command
            if current_word + char == "!exit":
                exit_flag.set()
                print("\nExiting...")
                break
                
            # Handle space - send the current word to the queue
            if char == ' ':
                if current_word:
                    word_queue.put(current_word)
                    pending_words.append(current_word)
                    print(f"\nAdded to queue: '{current_word}'")
                    current_word = ""
                sys.stdout.write("> ")
                sys.stdout.flush()
            # Handle enter key - also sends current word and starts a new line
            elif char == '\r' or char == '\n':
                if current_word:
                    word_queue.put(current_word)
                    pending_words.append(current_word)
                    print(f"\nAdded to queue: '{current_word}'")
                    current_word = ""
                sys.stdout.write("\n> ")
                sys.stdout.flush()
            # Add normal characters to the current word
            else:
                current_word += char
        
        except (EOFError, KeyboardInterrupt):
            exit_flag.set()
            break

def processing_thread():
    """Thread that processes words from the queue."""
    print("Processing thread started")
    
    while not exit_flag.is_set() or not word_queue.empty():
        try:
            # Try to get a word with a timeout to allow checking the exit flag
            word = word_queue.get(timeout=0.5)
            
            # Simulate processing time (200ms)
            time.sleep(0.2)
            
            # Remove from pending words list
            if word in pending_words:
                pending_words.remove(word)
            
            # Process the word (placeholder - replace with your processing logic)
            processed_word = word.upper()  # Simple example: convert to uppercase
            print(f"\nProcessed: '{word}' → '{processed_word}'")
            sys.stdout.write("> ")
            sys.stdout.flush()
            
            # Mark task as done
            word_queue.task_done()
            
        except queue.Empty:
            # No items in queue, just continue and check exit flag
            continue

def main():
    # Create and start threads
    input_thread_obj = threading.Thread(target=input_thread)
    processing_thread_obj = threading.Thread(target=processing_thread)
    
    input_thread_obj.daemon = True
    processing_thread_obj.daemon = True
    
    input_thread_obj.start()
    processing_thread_obj.start()
    
    # Wait for the exit flag
    while not exit_flag.is_set():
        time.sleep(0.1)
    
    print("Shutting down...")
    
    # Wait for threads to complete
    input_thread_obj.join(timeout=1)
    processing_thread_obj.join(timeout=1)
    
    print("Program terminated")

if __name__ == "__main__":
    main()
