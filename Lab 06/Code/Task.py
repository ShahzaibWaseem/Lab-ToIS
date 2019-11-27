import nltk   
from bs4 import BeautifulSoup
from urllib.request import urlopen

url = "https://www.w3.org/TR/PNG/iso_8859-1.txt"    
html = urlopen(url).read()    
raw = BeautifulSoup(html).get_text()
tokens = nltk.word_tokenize(raw) 
print("Tokens: ", tokens)