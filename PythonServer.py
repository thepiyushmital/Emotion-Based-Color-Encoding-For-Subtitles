# import socket programming library
import socket

# import thread module
from _thread import *
import threading
from colorama import init,Fore



#print_lock = threading.Lock()

colordict =	{
  "A++": "Fore.BLUE",
  "A+": "Fore.CYAN",
  "A-": "Fore.GREEN",
  "A": "Fore.GREEN",
  "B+": "Fore.GREEN",
  "B": "Fore.YELLOW",
  "B-": "Fore.RED",
  "B--": "Fore.BLACK"
}
# thread fuction
def threaded(c):
    while True:

        # data received from client
        data = c.recv(1024)
        data=data.decode('utf-8')
        #if data == 'exit':
            #print('Bye')

            # lock released on exit
            #print_lock.release()
            #break

        # reverse the given string from client
        #data = data[::-1]
        print(data)
        '''
        name,time,msg,color=data.split(';;')
		init(autoreset=True)
        if(color=="A++"):
            print(Fore.BLUE+name+":"+msg)
        elif(color=="A+"):
            print(Fore.CYAN+name+":"+msg)
        elif (color == "A-"):
            print(Fore.GREEN + name + ":" + msg)
        elif (color == "A"):
            print(Fore.GREEN + name + ":" + msg)
        elif (color == "B+"):
            print(Fore.GREEN + name + ":" + msg)
        elif (color == "B"):
            print(Fore.YELLOW + name + ":" + msg)
        elif (color == "B-"):
            print(Fore.RED + name + ":" + msg)
        elif (color == "B--"):
            print(Fore.WHITE + name + ":" + msg)
        # send back reversed string to client
        #c.send(data)
        '''

    # connection closed
        try:
            data=data.encode('utf-8')
            c.send(data)
        except socket.error:
            break
    c.close()


def Main():
    host = ""

    # reverse a port on your computer
    # in our case it is 12345 but it
    # can be anything
    port = 2007
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    #print("socket binded to post", port)

    # put the socket into listening mode
    s.listen(5)
    #print("socket is listening")

    # a forever loop until client wants to exit
    while True:
        # establish connection with client
        c, addr = s.accept()

        # lock acquired by client
        #print_lock.acquire()
        #print('Connected to :', addr[0], ':', addr[1])

        # Start a new thread and return its identifier
        start_new_thread(threaded, (c,))
    s.close()


if __name__ == '__main__':
    Main()
