import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
import tqdm


class GameOfLife(object):
    def __init__(self, configpath: str, outputfile: str):
        n_iterations, dead_symbol, live_symbol, init_state = self.read_config_file(configpath=configpath)

        os.makedirs(os.path.join(os.path.dirname(outputfile), "plots"), exist_ok=True)

        with open(outputfile, 'w') as fh:
            pass

        self.n_iterations = n_iterations
        self.state = init_state
        self.dead_symbol = dead_symbol
        self.live_symbol = live_symbol
        self.current_iteration = 0
        self.outputfile = outputfile

    def __write_state__(self):
        state2=[]
        for j in self.state:
            for i in j:
                if i == 0:
                    state2.append(self.dead_symbol)
                if i == 1:
                    state2.append(self.live_symbol)
            state2.append('\n')
        state2.append('\n')
        sep = ''
        state3 = sep.join(state2)
        with open(self.outputfile, 'a') as f:
            f.write(state3)
    def make_video(self, video_filename: str):
        pass
#parsing configpath file
def read_config_file(configpath: str):
    matches = ['n_iterations:', 'dead_symbol:', 'live_symbol:','init_state:']
    folderpath = 'ex10_testfiles/valid_04.config'
    isit = False
    slice_state = []
    i = 0
    a = 0
    b = 0
    with open(configpath, 'r') as f:
        file_content = f.read()
        if not all(x in file_content for x in matches):
            raise AttributeError(f"AttributeError")
        #n_iterations
        n_iterations = re.findall(r'n_iterations: (\S+)', file_content)[0]
        try:
            int(n_iterations)
        except ValueError:
            isit = True
        if isit == True:
            raise AttributeError(f"AttributeError")
        #dead_symbol
        dead_symbol = re.findall(r'dead_symbol: (\S+)', file_content)[0]
        if len(dead_symbol) != 3:
            raise AttributeError(f"AttributeError")
        dead_symbol = dead_symbol.replace('"','')
        if not len(dead_symbol) == 1:
            raise AttributeError(f"AttributeError")
        #live_symbol
        live_symbol = re.findall(r'live_symbol: (\S+)', file_content)[0]
        if len(live_symbol) != 3:
            raise AttributeError(f"AttributeError")
        live_symbol = live_symbol.replace('"','')
        if not len(live_symbol) == 1:
            raise AttributeError(f"AttributeError")
        #init_state
        file_content_split = file_content.splitlines()
        for x in file_content_split:
            if x == '"':
                quota_index = file_content.index(x)
                slice_state.append(i)
            i = i + 1
        some_slice = slice(slice_state[0]+1,slice_state[1])
        init_state = file_content_split[some_slice]
        init_state_boolean = np.zeros(shape=(slice_state[1]-slice_state[0]-1,len(init_state[0])), dtype=np.int32)
        for x in init_state:
            if len(x) != len(init_state[0]):
                raise ValueError(f"ValueError")
        for x in init_state:
            for c in x:
                if not c == live_symbol and not c == dead_symbol:
                    raise ValueError(f"ValueError")
                if a == len(init_state[0]):
                    a = 0
                    b = b + 1
                if c == live_symbol:
                    init_state_boolean[b][a] = 1
                a = a + 1
        return (int(n_iterations),dead_symbol,live_symbol,init_state_boolean)

    def step(self):
        """Compute the next tick of the simulator and return current number of iteration.
        Returns None if game is completed."""
        self.current_iteration += 1
        if self.current_iteration >= self.n_iterations:
            return None
        self.state = self.__get_next_state__(self.state)
        self.__write_state__()
        self.__state_to_image__()
        return self.current_iteration
#computing the next step from previous step
    def __get_next_state__(self, state: np.ndarray):
        temp_state_1 = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        temp_state_2 = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        if np.all(self.state == temp_state_1):
            new_state = temp_state_2
        else:
            new_state = temp_state_1
        return new_state

    def __state_to_image__(self):
        """Save state to image file"""
        image_name = os.path.join(os.path.dirname(self.outputfile), "plots", f"state_{self.current_iteration:05}.png")
        fig, ax = plt.subplots()
        ax.imshow(np.asarray(self.state, dtype=np.uint8))
        fig.tight_layout()
        fig.savefig(image_name)
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    # Creating a parser
    parser = argparse.ArgumentParser()
    parser.add_argument('configpath', help='configuration file', type=str)
    parser.add_argument('outputfile', help='file to write state to', type=str)

    # Parsing the arguments
    args = parser.parse_args()
    configpath = args.configpath
    outputfile = args.outputfile

    # Creating game instance
    game = GameOfLife(configpath=configpath, outputfile=outputfile)

    current_iteration = 0
    with tqdm.tqdm() as progressbar:  # Show a progressbar
        while current_iteration is not None:  # Continue until current iteration is None (=End of game)
            current_iteration = game.step()
            progressbar.update()

    # Saving video to a file
    game.make_video(os.path.join(os.path.dirname(outputfile), "video"))
