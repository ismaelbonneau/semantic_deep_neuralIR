# Petite explication des données 

chemin d'accès : (machine elephanz )   /local/karmim/Stage_M1_RI/data
ou               (machine toulouse )   /local/soulier/data-gia/Robust4/
Mais dans mon compte ( karmim ) j'ai juste pris les données qu'on avait besoin. 
pour se co : 

ssh user@gate.lip6.fr
puis 
ssh machine

## Dossier collection**
Pour le moment on va travailler sur la collection générique ( en gros documents de pleins de thèmes différents... ) Robust4.

Tout les docs se trouvent dans collection**, c'est des fichiers XML avec la balise <docno> qui désigne l'ID du doc. 

Normalement on a pas besoin d'indéxer toute la base, ça a déjà était fait avec le code d'adrien. 
http://big18:9200/

Le lien ici sert à accéder au moteur de recherche, et donc à la base indéxé construite avec le code d'adrien.
Pour y accéder il faut être sur une des machines du lip6 et taper l'url. ça va être utile dans notre code dans le fichier src/ir_engine.py . 
--> Normalement c'est indexé c'est tout bon. Je ferai les test dessus avec elasticsearch pour vérifier que tout est ok . 



## annotated_collection_tagme_score

### 015**
Dans ce dossier on a les mêmes dossiers/documents que dans collection**. Sauf qu'en fait ici, on ajoute ( dès qu'il existe ) le concept associé à ce mot. 
Par exemple la phrase : 
"POLITICIANS,  PARTY PREFERENCES "
va devenir
"politicians $#!Politics party $#!Political_party preferences". 
Donc la on a les concepts sous la forme $#!concept
pour le mot "preferences" on a pas de concept donc on a rien. 
Après je sais pas encore si on peut avoir une liste de concept, j'ai pas vu dans les données. 

### 015.relation**

Dans ce dossier on a les fichiers avec tout les concepts reliés directement entre eux, une ligne(assez longue) est une liste de tout les concepts directement relié du genre : 

$#!Government|$#!Provisional_Government_of_the_Republic_of_China_(1912)|$#!Indiana_University_Student_Association|$#!Central_government|...

Séparé par le caractère '|'



## qrels.robust2004**.txt
format 

id_query osef id_doc bool_pertinence 

en gros pour chaque query on te dit si pour ce doc il est pertinent ou non ( 1 si pertinent 0 sinon). 

##  robust2004_qrel.txt
je crois que c'est le même fichier que qrel.robust2004 mais à verifier. 


##  robust2004.txt 

EN GROS : 
Pour chaque query on a : 
	- la query en mot clé du genre : "president usa 2004".
	- la query en langage naturel : " qui était le président des états unis en 2004".

C'est toutes les query mises sous forme de dico. 
{id_query : (query_en_mot_clé, query_en_lang_nat),...}

## topic-robust4.xml 

C'est les mêmes query que robust2004 sauf qu'elles sont pas pré-traité, elles sont au format xml. 


## topics-title.annotated**.csv

C'est les query mis en format csv, sauf que la on a ajouté les concepts dès qu'ils existent. Et on a uniquement fait ça pour les query mot clé. Les concepts sont mis au même format que pour les docs. 

## topics-title.bar**.csv 

C'est les query sous format 

id_query , query_mot_clé 














