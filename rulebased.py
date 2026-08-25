responses = {
    "hello": "Hi there!",   
    "bye": "Goodbye!"
}

exit_commands = {"bye", "exit"}

while True:                           
    clean = raw.lower().strip()       

    if clean in exit_commands:         
        print("Bye!")
        break                        

    reply = responses.get(clean, "I do not understand.")  # step 5
    print(reply)                         
