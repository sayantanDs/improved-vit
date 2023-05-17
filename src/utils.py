import os
import shutil
import tensorflow as tf



def get_prev_save_file_name(model_save_loc):
    prev_save_file=""

    save_files = os.listdir(model_save_loc)
    save_files = [k for k in save_files if (k[-2:]=="h5" and k[-7:-3]!="best")]
    if len(save_files)>=1:    
        save_files.sort(reverse=True)
        prev_save_file = save_files[0]

    return prev_save_file


def get_prev_best_save_file_name(model_save_loc):
    prev_best_file = ""

    save_files = os.listdir(model_save_loc)
    best_save_files = [k for k in save_files if k[-7:]=="best.h5"]
    if len(best_save_files)>=1:    
        best_save_files.sort(reverse=True)
        prev_best_file = best_save_files[0]
    
    return prev_best_file



class CustomModelCheckPoint(tf.keras.callbacks.Callback):
    def __init__(self, model_save_loc, prev_save_file="", prev_best_file="", prev_best_acc=0, model_name="epoch", **kargs):
        super(CustomModelCheckPoint,self).__init__(**kargs)
        self.model_save_loc = model_save_loc
        self.model_name = model_name
        self.prev_save_file = prev_save_file
        self.prev_best_file = prev_best_file
        self.prev_best_acc = prev_best_acc

    def on_epoch_end(self, epoch, logs={}):
        # acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")

        filename =  f"{self.model_name}_{(epoch+1):03d}-{val_acc:.3f}.h5"
        self.model.save_weights(os.path.join(self.model_save_loc, filename)) # save the model
        
        # remove previous epoch save files
        if self.prev_save_file:
            delete_filename = os.path.join(self.model_save_loc, self.prev_save_file)
            open(delete_filename, 'w').close() # overwrite and make the file blank
            os.remove(delete_filename)
        self.prev_save_file = filename

        # save best model till now
        if val_acc > self.prev_best_acc:           
            best_filename = filename[:-3]+"_best.h5"
            shutil.copy(os.path.join(self.model_save_loc, filename), os.path.join(self.model_save_loc, best_filename))
           
            if self.prev_best_file:
                delete_filename = os.path.join(self.model_save_loc, self.prev_best_file)
                open(delete_filename, 'w').close() # overwrite and make the file blank
                os.remove(delete_filename)
           
            self.prev_best_acc = val_acc
            self.prev_best_file = best_filename  
