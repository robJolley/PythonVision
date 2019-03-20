#print('Enter Table:')
#userInput = input()

#try:
#	value = int(userInput)
#	if(value > 0):
#		for x in range(0,13,1):
#			sum = x*value
#			print (value,' by ',x,' = ',sum)
		
#except ValueError:
#	print('Thats not a number silly')

import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
for x in range (37,10000,50):
	print ('Roisin Rocks :', x)
	winsound.Beep(x, duration)