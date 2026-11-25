.PHONY: setup data data-java train
setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
data:
	python3 simulation/python/pen_simulator.py --out data/processed/synthetic_python.csv.gz --seed 11
data-java:
	javac -d /tmp/pen simulation/java/PenEngine.java && java -cp /tmp/pen PenEngine data/processed/synthetic_java.csv 11
train:
	python3 training/train_starter.py data/processed/synthetic_python.csv.gz training/out
