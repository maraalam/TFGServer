
from transformers.optimization import  Adafactor 
from transformers import T5Tokenizer, T5ForConditionalGeneration
import warnings
import torch
import warnings
from IPython.display import HTML, display
import matplotlib.pyplot as plt

class Model:
    
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
        self.__DEFAULT_PARAMS = {"BATCH_SIZE" : 4, "NUM_OF_EPOCHS" : 2}

        if torch.cuda.is_available():
            self.dev = torch.device("cuda:0") 
            print("Running on the GPU")
        else:
            self.dev = torch.device("cpu")
            print("Running on the CPU")
        
        self.model.to(self.dev)
        self.optimizer = Adafactor(
            self.model.parameters(),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

        self.global_loss_per_10_steps = []
        self.last_train_loss_per_10_steps = []
    
    def train(self, train_df, batch_size = None, num_of_epochs = None, display_progress = 0):

        batch_size, num_of_epochs =  self.__checkTrainParams(batch_size, num_of_epochs)

        num_of_batches = len(train_df) / batch_size

        num_of_batches = int(num_of_batches)

        #Sets the module in training mode
        self.model.train()

        self.last_train_loss_per_10_steps=[]
        for epoch in range(1,num_of_epochs+1):
            print('Running epoch: {}'.format(epoch))
            
            running_loss=0

            if(display_progress): out = display(self.__progress(1, num_of_batches+1), display_id=True)
            
            for i in range(num_of_batches):
                inputbatch=[]
                labelbatch=[]
                new_df=train_df[i*batch_size:i*batch_size+batch_size]

                for indx,row in new_df.iterrows():
                    input = row['input_text'] + self.tokenizer.eos_token
                    labels = row['target_text'] + self.tokenizer.eos_token   
                    inputbatch.append(input)
                    labelbatch.append(labels)

                inputbatch=self.tokenizer.batch_encode_plus(inputbatch,padding=True,max_length=400,return_tensors='pt')["input_ids"]
                labelbatch=self.tokenizer.batch_encode_plus(labelbatch,padding=True,max_length=400,return_tensors="pt") ["input_ids"]
                inputbatch=inputbatch.to(self.dev)
                labelbatch=labelbatch.to(self.dev)

                # clear out the gradients of all Variables 
                self.optimizer.zero_grad()

                # Forward propogation
                outputs = self.model(input_ids=inputbatch, labels=labelbatch)
                loss = outputs.loss
                loss_num=loss.item()
                logits = outputs.logits
                running_loss+=loss_num

                if i%10 ==0:    
                    self.global_loss_per_10_steps.append(loss_num)  
                    self.last_train_loss_per_10_steps.append(loss_num)

                if(display_progress): out.update(self.__progress(loss_num,i, num_of_batches+1))

                # calculating the gradients
                loss.backward()

                #updating the params
                self.optimizer.step()
                
            running_loss=running_loss/int(num_of_batches)
            print('Epoch: {} , Running loss: {}'.format(epoch,running_loss))
            

    def encode(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        return input_ids

    def generateText(self, encode_text, do_sample = False, num_beams = 5, num_beam_groups=1,no_repeat_ngram_size = 2,
     min_length = 0, max_length = 500, top_k = 50, top_p = 0.95, temperature = 1.0, penalty = 1.0, num_return_sequences = 1, early_stopping=True):
        
        #Sets the module in evaluation mode
        self.model.eval()
        encode_text = encode_text.to(self.dev)

        outputs = self.model.generate(encode_text)

        outputs = self.model.generate(encode_text, 
                    do_sample = do_sample,    #Si false devuelve distintas frases
                    num_beams = num_beams,
                    no_repeat_ngram_size = no_repeat_ngram_size, #If set to int > 0, all ngrams of that size can only occur once.
                    num_beam_groups = num_beam_groups,
                    min_length = min_length,
                    max_length = max_length,
                    top_k = top_k,            #The number of highest probability vocabulary tokens to keep for top-k-filtering.                          
                    top_p = float(top_p),          #If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
                    temperature = float(temperature),
                    repetition_penalty = penalty,#1.0 No penalty
                    num_return_sequences = num_return_sequences,
                    early_stopping = early_stopping)

        return outputs

    
    def decode(self, encode_text):
        result = self.tokenizer.decode(encode_text)
        return result
        

    def plot_global_loss(self):
        steps = [i*100 for i in range(len(self.global_loss_per_10_steps))]
    
        plt.plot(steps, self.global_loss_per_10_steps)
        plt.title('Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.show()


    def plot_last_loss(self):
        steps = [i*100 for i in range(len(self.last_train_loss_per_10_steps))]
    
        plt.plot(steps, self.last_train_loss_per_10_steps)
        plt.title('Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.show()


    def save_model(self, target_url):
        self.model.save_pretrained(target_url)


    def load_model(self, source_url):
        self.model = self.model.from_pretrained(source_url)
        

    def __progress(loss,value, max=100):
        return HTML(""" Batch loss :{loss}
            <progress
                value='{value}'
                max='{max}',
                style='width: 100%'
            >
                {value}
            </progress>
        """.format(loss=loss,value=value, max=max))


    def __checkTrainParams(self, batch_size, num_of_epochs):
        if batch_size is None:
            batch_size = self.__DEFAULT_PARAMS["BATCH_SIZE"]

        if num_of_epochs is None:
            num_of_epochs = self.__DEFAULT_PARAMS["NUM_OF_EPOCHS"]


        return batch_size, num_of_epochs

