#gym 
import gymnasium as gym
from time import sleep
#torch
from torch import nn
import torch
from torch.utils.data import DataLoader
#funzioni matematiche
from numpy import sin, cos
import numpy as np
from math import ceil
import math
import matplotlib.pyplot as plt
import random
#funzioni nostre
from mem_buffer import MemBuffer

# LdSR: this policy is not a policy, you are magically setting the state to a random position. -1 project 
def policy(env): #creo la funzione policy per campionare lo spazio in modo omogeneo 

    angolo1 = random.uniform(-math.pi, math.pi) #generiamo angoli in modo casuale tra -180 e +180
    angolo2 = random.uniform(-math.pi, math.pi) #in radianti 

    env.unwrapped.state = np.array([angolo1, angolo2])

def get_angular(env):
    state_angles = env.unwrapped.state
    normalised_state_angles = state_angles / math.pi
    return torch.tensor(normalised_state_angles, dtype=torch.float)

def get_cartesian(env):
    s = env.unwrapped.state
    p1 = [
        -env.unwrapped.LINK_LENGTH_1 * cos(s[0]),
        env.unwrapped.LINK_LENGTH_1 * sin(s[0]),
        ]
    p2 = [
        p1[0] - env.unwrapped.LINK_LENGTH_2 * cos(s[0] + s[1]),
        p1[1] + env.unwrapped.LINK_LENGTH_2 * sin(s[0] + s[1]),
        ]
    # LdSR: should be both angles. -1 project    
    return torch.tensor(p2, dtype=torch.float)      #voglio solo 2 output

# add parameters for number of layers, layer size
class NN(nn.Module):
    def __init__(self, env, n_layers=2, layer_size=4, lr=0.001):
        
        nn.Module.__init__(self)
        # add env
        self.env = env

        # add n_layers, layer_size, lr, (n_iter, max_epochs)
        self.n_layers = n_layers
        self.layer_size = layer_size

        # NN 
        # self.nn = nn.Sequential(
        #        nn.Linear(2, 4),
        #        nn.ReLU(),
        #        nn.Linear(4, 2),
        #        nn.ReLU(),
        #        )
        #commento per avere un'architettura della rete dinamica e non fissa
        
        #architettura dinamica della NN
        layers = []
        layers.append(nn.Linear(2, layer_size))         #primo layer (ingresso)
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):                   #layer intermedi (nascosti); per avere il giusto numero di iterazioni del ciclo for
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_size, 2))         #ultimo layer (uscita)
        self.nn = nn.Sequential(*layers)

        # LdSR: training artifacts should not be part of the model. -1 code quality
        #Optimizer and Loss Function choice
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr = lr)
        self.train_losses = []
        self.val_losses = []

        return None

    def forward(self, x):
        return self.nn(x)

    # -----------------------------------------------------
    # training functions

    # add train iter
    def train_iter(self, angles, target_coordinates) -> list :    
        pred_coordinates = self.nn(angles)
        loss = self.loss_function(target_coordinates, pred_coordinates)
        
        # LdSR: parameter update should be done each epoch, not each iteration. -1 project
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # add train epoch
    def train_epoch(self, n_epoch, train_loader, validation_loader):
        self.train_losses = []
        for epoch in range(n_epoch):
            self.nn.train()                            #per riavere la rete in modalità allenamento dopo aver chiamato validation
            total_train_loss = 0
            for _, (angles, target_coordinates) in enumerate(train_loader):
                self.optimizer.zero_grad()
                loss_value = self.train_iter(angles, target_coordinates)
                total_train_loss += loss_value
            epoch_avg_losses = total_train_loss/len(train_loader)
            self.train_losses.append(epoch_avg_losses)
            val_loss = self.validation(validation_loader)
            print(f"Epoch (Epoca): {epoch + 1}/{n_epoch} - Train Loss: {epoch_avg_losses:.6f} - Val Loss: {val_loss:.6f}")
        return self.train_losses, self.val_losses

    #aggiunta funzione per la validazione alla fine di ogni epoca
    def validation(self, dataloader):
        self.nn.eval()                                #per avere la rete in modalità valutazione
        total_val_loss = 0
        with torch.no_grad():                         #per risparmiare memoria non calcolo il gradiente  
            for angles, coords_target in dataloader:
                pred_coords = self.nn(angles)          #calcola le predizioni
                loss = self.loss_function(pred_coords, coords_target)           #calcola la loss e confronta le predizioni con i target
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(dataloader)                         #media della loss sul validation_dataset
        self.val_losses.append(avg_val_loss)
        return avg_val_loss

    # add fn to print training and val losses
    def plot_losses(self):
        plt.plot(self.train_losses, label='Train_loss')
        plt.plot(self.val_losses, label='Val-loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Andamento della Val_loss in confronto alla Train_loss")
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()
        return None


#-----------------------------------------------------------------------


if __name__ == '__main__':
    #aggiunta del seed per avere sempre una stessa inizializzazione
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # control params 
    max_it = 5000 #aumento a 5000 per avere un dataset decente 
    batch_size = 32 #dimensione del batch per l'allenamento 

    env = gym.make('Acrobot-v1') #ho rimosso  " render_mode='human'" come consigliato dal prof 
                                    #per velocizzare la raccolta di dati (quel comando è per l'animazione)
    buffer_training = MemBuffer(max_size= max_it) #ho inizializzato il buffer
    buffer_validation = MemBuffer (max_size= ceil(max_it*0.2)) #ho inizializzato il buffer

    episode_over = False

    env.reset(seed=42)
    
    # LdSR: should be adding data each epoch, not creating the buffer in the beginning. -1 project
    #it = 0
    #while (not episode_over) and (it < max_it):
    for i in range(max_it): 
        #action = env.action_space.sample()  # agent policy that uses the observation and info
        action = policy(env)
        #observation, reward, terminated, truncated, info = env.step(action)

        angles = get_angular(env)
        goal = get_cartesian(env)

        buffer_training.add(angles, goal) #inseriamo angles e goal all'interno del buffer creato sopra

        


        #episode_over = terminated or truncated
        #it += 1
        # sleep(0.15) lo commento perchè se no ci mette una vita a runnare 
        

    for i in range(ceil(max_it*0.2)): 
        #action = env.action_space.sample()  # agent policy that uses the observation and info
        action= policy(env)
        #observation, reward, terminated, truncated, info = env.step(action)

        angles = get_angular(env)
        goal = get_cartesian(env)

        buffer_validation.add(angles, goal) #inseriamo angles e goal all'interno del buffer creato sopra

        


        #episode_over = terminated or truncated
        #it += 1
        # sleep(0.15) lo commento perchè se no ci mette una vita a runnare 
        
    env.close()
    training_loader = DataLoader(dataset = buffer_training, batch_size = batch_size, shuffle = True  )
    validation_loader = DataLoader(dataset = buffer_validation, batch_size = batch_size, shuffle = False )

    print(f"dati raccolti nel buffer training :  {len(buffer_training)}")
    print(f"dati raccolti nel buffer validation :  {len(buffer_validation)}")


#PLOT PER VEDERE L'OMOGENEITA' DEI DATI
    angolo1 = [t[0].item() for t in buffer_training.x]
    angolo2 = [t[1].item() for t in buffer_training.x]
    
    x = [t[0].item() for t in buffer_training.y]
    y = [t[1].item() for t in buffer_training.y]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    plot_angoli = ax1.hist2d(angolo1, angolo2, bins=30, cmap='viridis')
    ax1.set_title('Densità degli Angoli')
    ax1.set_xlabel(r'$\theta$1')
    ax1.set_ylabel(r'$\theta$2')
    fig.colorbar(plot_angoli[3], ax=ax1, label='Numero di campioni per area')

    plot_posizioni = ax2.hist2d(x, y, bins=30, cmap='viridis')
    ax2.set_title('Densità dello Spazio Cartesiano')
    ax2.set_xlabel('Coordinata X')
    ax2.set_ylabel('Coordinata Y')
    ax2.axis('equal')
    fig.colorbar(plot_posizioni[3], ax=ax2, label='Numero di campioni per area')

    plt.tight_layout()
    plt.show()

############### LdSR: code outside main -1 code quality
#tuning con grid search
#griglia
grid_layers = [1, 2, 3]
grid_neurons = [16, 64, 128]
grid_lr = [0.00001, 0.0001, 0.001, 0.01]

best_val_loss = float('inf')
best_config = None
best_model = None

configs = []                #lista che salva la configurazione e l'ultimo valore della val_loss per trovare la migliore
best_global_configs = []    #lista che salva la configurazione migliore della rete, ogni volta che se ne trova una migliore,
                            #per fare un confronto finale di alcune curve di val_loss per vedere qual è la migliore anche graficamente

#tuning
# LdSR: good analysis. +1 project
for l in grid_layers:
    for n in grid_neurons:
        for lr in grid_lr:
            print(f"\n Numero di livelli test: {l}, numero di neuroni in ogni livello test: {n}, learning rate test: {lr}")
            model = NN(env, n_layers=l, layer_size=n, lr=lr)
            train_loss, val_loss = model.train_epoch(n_epoch=60, train_loader=training_loader, validation_loader=validation_loader)
            current_config = {
                'configurazione testata': f"numero di layer: {l}, 'numero di neuroni: {n}, 'lr: {lr}",
                'last_value_val_loss': val_loss[-1]                 #performance finale del modello al termine dell'ultima epoca
            }
            configs.append(current_config)
            if current_config['last_value_val_loss'] < best_val_loss:
                best_val_loss = current_config['last_value_val_loss']
                best_config = {
                    'numero di layer': l,
                    'numero di neuroni': n,
                    'learning rate (lr)': lr
                }
                best_layer_config = {
                    'configurazione testata': f"numero di layer: {l}, 'numero di neuroni: {n}, 'lr: {lr}",
                    'val_loss': val_loss                            #salva la curva della val_loss per poi fare il plot
                }
                best_global_configs.append(best_layer_config)
                best_model = model
print(f"\n La configurazione migliore e': {best_config} e la validation loss ottenuta è {best_val_loss:.6f}")
#torch.save(model.state_dict(), 'best_model_config.pth')           #salva il modello migliore della rete se si vuole usare in futuro
#la precedente riga è commentata perché non vogliamo che il file venga erroneamente carivato su github

#plot della train loss e della val loss del modello migliore
best_model.plot_losses()

#plot per confrontare le diverse val losses
plt.figure(figsize=(10, 6))
for config in best_global_configs:
    val_loss_curve_test = config['val_loss']
    label_curve = config['configurazione testata']
    plt.plot(val_loss_curve_test, label=label_curve)
plt.xlabel("Epoche")
plt.ylabel("Val_loss")
plt.title("Confronto delle Validation Losses")
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()
