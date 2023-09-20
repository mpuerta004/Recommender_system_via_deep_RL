# Vamos a intentar no hacer embeddings primero, vemos como va el sistema y luego si veos qu eva lo intentamos con embeddings

#Entonces el embedding lo que hace es regularizar los datos, por tanto los redimenciona que es algo que nosotro sno tenemos redimencionado 
#Tenemos que pensar la reprsentacion de los datos. 
# Tenemos que generar una representacion numero de los datos que tenemos.

from models.Campaign_Member import Campaign_Member
from models.Hive_Member import Hive_Member

#The role of the user depends of the campaigns in where the cell belongs
"""
    We return 0.0 is the role is one in where the recomendad doents has to recomend anything 
    We return 0.5 if the role is WorkerBee
    We return 1.0 if the role is QueenBee
"""
def numeric_role(a: Campaign_Member ):
    #TODO! no se si sera texto o integrate
    if a.role == "WorkerBee":
        return 0.5 
    if a.role == "QueenBee":
        return 1.0
    if a.role == "BeeKeeper":
        return 0.0
    if a.role == "Hive":
        return 0.0
    if a.role =="DroneBee":
        return 0.0
    

"""
    We return 0.0 is the role is one in where the recomendad doents has to recomend anything 
    We return 0.5 if the role is WorkerBee
    We return 1.0 if the role is QueenBee
"""
def numeric_role(a:Hive_Member):
    #TODO! no se si sera texto o integrate
    if a.role == "WorkerBee":
        return 0.5 
    if a.role == "QueenBee":
        return 1.0
    if a.role == "BeeKeeper":
        return 0.0
    if a.role == "Hive":
        return 0.0
    if a.role =="DroneBee":
        return 0.0



########################## Metadatos del usuario ############################################


#Los metadatos que hay que trasladar son: 
# genero 
# Edad 




