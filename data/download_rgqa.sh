# Download rgqa data from https://drive.google.com/file/d/1p5GsYo80blyxM1TYGr0AOIjgo5DFFpbU/view?usp=share_link 
gdown https://drive.google.com/uc?id=1p5GsYo80blyxM1TYGr0AOIjgo5DFFpbU
unzip gqa.zip 
rm gqa.zip

# download butd data
# https://drive.google.com/file/d/13YJFfcJsZnIF-pAl-WWYiHuWPleyS24t/view?usp=sharing
gdown https://drive.google.com/uc?id=13YJFfcJsZnIF-pAl-WWYiHuWPleyS24t
unzip butd.zip
rm butd.zip

wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.300d.txt butd/
rm glove*


