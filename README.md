### Model in podatki

Model je Graph Neural Network, specifično Pinsage:

<https://arxiv.org/pdf/1806.01973.pdf>

![1641838776410](C:\Users\thema\AppData\Roaming\Typora\typora-user-images\1641838776410.png)

Vsakemu vozlišču v grafu pripišemo značilke. Algoritem izvede konvolucijo nad vsakim vozliščem i.e. agregira značilke sosedov in danega vozlišča v končni embeding. Sosedi niso direktni sosedi iz grafa ampak pridobljeni z Personalized PageRank. Pinsage specifično je induktiven in ne deluje na celotni matriki grafa ampak dinamično generira batche kot je prikazano.



Apliciran je na glasbo, i.e. generiranje embedingov komadov za namene priporočilnega sistema (podobni komadi so bližnji vektorji).

Dataset je bipartidni graf komadov in seznamov predvajanja, kjer povezava pomeni pripadnost komada nekemu seznamu predvajanja.
Značilke vozlišč so audio embeddingi na osnovi zvočnega posnetka komada, specifično OpenL3 model.

<img src="C:\Users\thema\AppData\Roaming\Typora\typora-user-images\1641839533621.png" width=400px>

### Učenje

Učni podatki (ground truth) so trojice oblike *<query, positive, negative>* kjer sta *query* in *positive* par komadov, za katera vemo, da sta podobna, *negative* pa jima ni podoben (npr. naključni komad).

Izgubna funkcija (max-margin loss) kaznuje preslikavo, kjer sta *query* in *negative* bolj podobna kot *query* in *positive*.

![1641840471629](C:\Users\thema\AppData\Roaming\Typora\typora-user-images\1641840471629.png)



### Problem

Trenutno testiram učenje algoritma na majhni podmnožici dataseta: cca. **5000** vozlišč (komadov) in cca. 5000 učnih trojic. 

Trenutno so *<query, positive>* pari dobljeni preko Personalized PageRank podobnosti v grafu.

Batch je zgrajen z naključnim vzorčenjem teh parov in dodajanjem *negative* vozlišča.



Učenje se zdi zelo nestabilno.

Z pravilno izbiro parametrov mi sicer uspe priti do lepe izgubne krivulje in nekega minimuma. 

![2022-01-10 19_54_18](C:\Users\thema\Desktop\Snap\2022-01-10 19_54_18.png)

Vendar pa na končnih evalvacijskih metodah (hit-rate, mean reciprocal rank), dobljeni embedingi še vedno dajejo slabši rezultat kot izključno audio embeddingi i.e. značilke vozlišč, kljub temu, da

	1. Algoritem dobi te značilke kot vhod
 	2. Glede na število učnih primerov in parametrov, bi pričakoval overfitting



Učni parametri so (za "modro črto"): 

lr = 1e-3

batch_size = 128

50 batches per epoch

10 epochs

in_dim = 512 (node features)

hidden_dim = 512

out_dim = 128

exp. lr cooldown (k = 0.9)



Model spisan v *pinsage_model.py* in *pinsage_training.py* . Evalvacija v *eval.py* .



Razložim bolj podrobno. Ubistvu me bolj kot ne zanima mnenje o postopku učenja, da ne bom delal kakšnih neumnosti.

LP :)