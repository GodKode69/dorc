from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CLASSES = [
    "alpaca", "anteater", "antelope", "armadillo", "baboon", "badger", "bat",
    "bear", "bee", "beetle", "bison", "blue_jay", "butterfly", "camel",
    "cardinal", "cat", "caterpillar", "chameleon", "cheetah", "chickadee",
    "chimpanzee", "cicada", "clams", "cockroach", "corals", "cow", "coyote",
    "crab", "crocodile", "crow", "deer", "dog", "dolphin", "donkey",
    "dragonfly", "duck", "eagle", "eel", "elephant", "egret", "finch", "fish",
    "flamingo", "fly", "fox", "frog", "gecko", "giraffe", "goat", "goldfish",
    "goose", "gorilla", "grasshopper", "groundhog", "hamster", "hare",
    "hedgehog", "heron", "hippopotamus", "hornbill", "horse", "hummingbird", "hyena",
    "ibis", "iguana", "jackal", "jellyfish", "kangaroo", "koala", "ladybugs",
    "leopard", "lion", "lizard", "lobster", "manatee", "mongoose", "moose",
    "mosquito", "moth", "mouse", "octopus", "okapi", "orangutan", "ostrich",
    "otter", "owl", "oyster", "panda", "parrot", "pelican", "penguin", "pig",
    "pigeon", "platypus", "porcupine", "possum", "puffers", "raccoon", "rat",
    "rhinoceros", "robin", "salamander", "sandpiper", "scorpion", "seahorse",
    "seal", "sea_rays", "seaslug", "sea_urchins", "shark", "sheep", "shrew",
    "shrimp", "sloth", "snail", "snake", "sparrow", "spider", "squid",
    "squirrel", "starfish", "starling", "sugar_glider", "swan", "tapir",
    "tiger", "toad", "turkey", "turtle", "vicuna", "walrus", "warbler",
    "weasel", "whale", "wolf", "wombat", "woodpecker", "wren", "yak",
    "zebra",
]
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.0003
TRAIN_SPLIT = 0.8
NUM_WORKERS = 4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 7
MIN_IMAGES = 100

MODEL_SAVE_PATH = BASE_DIR / "model.pt"
