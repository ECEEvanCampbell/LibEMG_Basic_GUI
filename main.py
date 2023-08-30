import libemg
from tkinter import *
import os

SUBJECT_NUMBER = 0
WINDOW_SIZE = 40
WINDOW_INC  = 10
FEATURES    = ["MAV","ZC","SSC","WL"]

class GUI:
    # lets set up some of the communication stuff here
    def __init__(self):
        # streamer initialization goes here.
        # I am using a myo, but you could sub in the delsys here
        self.streamer = libemg.streamers.myo_streamer()

        # create an online data handler to listen for the data
        self.odh = libemg.data_handler.OnlineDataHandler()
        # when we start listening we subscribe to the data the device is putting out
        self.odh.start_listening()

        # save_directory:
        self.save_directory = 'data/subject'+str(SUBJECT_NUMBER)+"/"
        if not os.path.isdir(self.save_directory):
            os.makedirs(self.save_directory)
        
        # make the gui
        self.initialize_ui()
        # hang
        self.window.mainloop()
        
    # lets set up some of the GUI stuff here
    def initialize_ui(self):
        # tkinter window (ugly but simple to code up)
        self.window = Tk()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.title("LibEMG GUI")
        self.window.geometry("500x300")

        # add some widgets to the gui

        # get training data button
        Button(self.window, font=("Arial", 10), text="Screen Guided Training", 
               command=self.launch_training).pack(pady=(0,10))
        # start signal visualization button
        #   Note: I'm not sure how nicely things play when data is being consumed in multiple regions
        #         i.e., visualizing signal, classifying on signal, etc.
        #         This is probably fine.
        Button(self.window, font=("Arial", 10), text="Visualize signal", 
               command=self.visualize_signal).pack(pady=(0,10))
        # start live classifier button
        Button(self.window, font=("Arial", 10), text="Visualize Feature Space", 
               command=self.visualize_feature_space).pack(pady=(0,10))
        # start live classifier button
        Button(self.window, font=("Arial", 10), text="Start Live Classifier", 
               command=self.start_classifier).pack(pady=(0,10))
        # visualize live classifier 
        Button(self.window, font=("Arial", 10), text="Visualize classifier", 
               command=self.visualize_classifier).pack(pady=(0,10))
        
    def launch_training(self):
        self.window.destroy()
        # get rid of GUI window in favour of libemg training gui
        training_ui = libemg.screen_guided_training.ScreenGuidedTraining()
        # you can find what these numbers in the list correspond to at:
        # https://github.com/LibEMG/LibEMGGestures
        # training will have all the classes you've downloaded into the images folder
        training_ui.download_gestures([1,2,3,4,5], "images/")
        training_ui.launch_training(self.odh, 3, 3, "images/", self.save_directory, 2)
        # the thread is blocked now until the training process is completed and closed 
        # once the training process ends, relaunch the GUI
        self.initialize_ui()
    
    def visualize_signal(self):
        self.window.destroy()
        self.odh.visualize()
        # ^ this blocks until its closed
        # v once closed it'll reopen the GUI
        self.initialize_ui()

    def visualize_feature_space(self):
        self.window.destroy()
        offlinedatahandler = self.get_data()
        windows, metadata  = self.extract_windows(offlinedatahandler)
        features = self.extract_features(windows)
        libemg.feature_extractor.FeatureExtractor().visualize_feature_space(features, "PCA", classes=metadata["classes"])
        # ^ this blocks until its closed
        # v once closed it'll reopen the GUI
        self.initialize_ui()


    
    def start_classifier(self):
        offlinedatahandler = self.get_data()
        windows, metadata  = self.extract_windows(offlinedatahandler)
        features = self.extract_features(windows)
        # we need to make an offline classifier to pass to the online classifier
        offlineclassifier = libemg.emg_classifier.EMGClassifier()
        feature_dictionary = {"training_features": features,
                              "training_labels"  : metadata["classes"]}
        offlineclassifier.fit("LDA", feature_dictionary=feature_dictionary)
        self.onlineclassifier = libemg.emg_classifier.OnlineEMGClassifier(offlineclassifier,
                                                                          WINDOW_SIZE,
                                                                          WINDOW_INC,
                                                                          self.odh,
                                                                          FEATURES,
                                                                          std_out=True,
                                                                          output_format="probabilities")
        # start running the online classifier in another thread (block=False)
        self.onlineclassifier.run(block=False)
        

    def visualize_classifier(self):
        self.window.destroy()
        self.onlineclassifier.visualize(legend=["Hand Closed", "Hand Open", "No Motion", "Wrist Extension", "Wrist Flexion"])
        self.initialize_ui()

    def get_data(self):
        classes_values = [str(i) for i in range(5)] # will only grab classes 0,1,2,3,4 currently
        reps_values    = [str(i) for i in range(3)] # will only grab reps 0,1,2 currently
        classes_regex  = libemg.utils.make_regex(left_bound="_C_",right_bound=".csv", values=classes_values)
        reps_regex     = libemg.utils.make_regex(left_bound="/R_", right_bound="_C_", values=reps_values)
        dic = {
            "classes": classes_values,
            "classes_regex": classes_regex,
            "reps": reps_values,
            "reps_regex": reps_regex
        }
        offlinedatahandler = libemg.data_handler.OfflineDataHandler()
        offlinedatahandler.get_data(folder_location=self.save_directory, filename_dic = dic, delimiter=',')
        return offlinedatahandler
    
    def extract_windows(self, offlinedatahandler):
        windows, metadata  = offlinedatahandler.parse_windows(WINDOW_SIZE, WINDOW_INC)
        return windows, metadata
    
    def extract_features(self, windows):
        features           = libemg.feature_extractor.FeatureExtractor().extract_features(FEATURES,
                                                                                         windows)
        return features
    
    # what happens when the GUI is destroyed
    def on_closing(self):
        # Clean up all the processes that have been started
        self.odh.stop_listening()
        self.window.destroy()


if __name__ == "__main__":
    gui = GUI()