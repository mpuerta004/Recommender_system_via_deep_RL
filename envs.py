import numpy as np

class OfflineEnv(object):
    #necesito saber como es ese users_dict
    def __init__(self, users_dict, users_history_lens, movies_id_to_movies, state_size, fix_user_id=None):

        self.users_dict = users_dict
        self.users_history_lens = users_history_lens # Es el cardinal de interacciones de ranking mayor o igual que 4 para cada usuario. el indice es el id del uaurio. 
        self.items_id_to_name = movies_id_to_movies
        
        self.state_size = state_size # numero de items que tiene el estado para generarlo... 
        self.available_users = self._generate_available_users()

        self.fix_user_id = fix_user_id
        
        # No tengo claro si van a darle un id a la info de cada usuario o que van a hacer la verdad... 
        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users)
        #Se selecciona uno al azar!!!! 
        
        #  Esto es para poder leer el users_dict -> no se si esto va a funcionar pero bueno... 
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]} 
        print("user_items", self.user_items)
        #{1210: 4, 245: 5, 1097: 4, 2915: 3, 2
        
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]] 
        print("ITems",self.items)
        #[1210,245,1097,2915, ...]
        #los ultimos 10 items que ha visto el usuario y que ha valorado! 
    
        self.done = False
        # TODO! esto hay que cambiarlo al otro! 
        self.recommended_items = set(self.items)
        # combierte los item en un set {}
        print("recommended_items", self.recommended_items)
        self.done_count = 3000
        
        
    def _generate_available_users(self):
        available_users = []
        for i, length in zip(self.users_dict.keys(), self.users_history_lens):
            #Seleccionas solo aquellos usuarios que tienen un historial mayor que el estado que quieres generar.
            if length > self.state_size:
                available_users.append(i)
        return available_users
    #devuelva la lista de id de usuarios cuyo historial es mayor que el estado que quieres generar. Pero no se si me gusta que se fije en el users_history_lens... porque ahi estan las que son positivas ( nota de 4 o mas)
    
     
    
    def _generative_available_items(self):
        available_items = []
        # for i in self.items_id_to_name.keys():
        #     if i not in self.user_items.keys():
        #         available_items.append(i)
        return available_items
    
    
    def reset(self):
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        return self.user, self.items, self.done
        
    def step(self, action, top_k=False):
        reward = -0.5
        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:
                #El nuevo item no esta entre los recomendados y esta en el historial del usuario (ya esta puntuado)
                if act in self.user_items.keys() and act not in self.recommended_items:
                    correctly_recommended.append(act) 
                    #No entiendo esto de -3 / 2  -> 
                    # si es 5 -> 1, si es 4 -> 0.5, si es 3 -> 0, si es 2 -> -0.5, si es 1 -> -1
                    rewards.append((self.user_items[act] - 3)/2)
                else:
                    rewards.append(-0.5)
                self.recommended_items.add(act)
            if max(rewards) > 0:
                # Si hay alguno que es mayor que 0, se añade al final de la lista de items y se quita el primero
                # Esto es porque tenemos el historial de items.. Entonces tenenos que ver cuales ha puntuado. 
                self.items = self.items[len(correctly_recommended):] + correctly_recommended
            reward = rewards
        else:
            if action in self.user_items.keys() and action not in self.recommended_items:
                reward = self.user_items[action] -3  # reward
            if reward > 0:
                self.items = self.items[1:] + [action]
            self.recommended_items.add(action)

        if len(self.recommended_items) > self.done_count or len(self.recommended_items) >= self.users_history_lens[self.user-1]:
            self.done = True
            
        return self.items, reward, self.done, self.recommended_items

    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.items_id_to_name[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names
