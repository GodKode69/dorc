from pathlib import Path

basePath = Path(__file__).resolve().parent.parent
dataDir = basePath / "data"
imgDir = basePath / "img"
newDir = imgDir / "new"
unconfirmedDir = imgDir / "unconfirmed"
untrainedDir = imgDir / "untrained"

classes = [
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
numClasses = len(classes)

imgSize = 224
batchSize = 384
epochs = 30
lr = 0.0003
warmupEpochs = 3
trainSplit = 0.8
numWorkers = 8
weightDecay = 0.01
labelSmoothing = 0.1
gradientClipNorm = 1.0
weightBlend = 0.5
mismatchPenalty = 0.1
retrainEpochs = 15
earlyStopPatience = 7
minImages = 100

modelSavePath = basePath / "model.pt"
