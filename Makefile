OS := $(shell uname -s)

ifeq ($(OS), Linux)
	ACTIVATE = . venv/bin/activate
else
	ACTIVATE = venv\Scripts\activate
endif

all:
	cd code && python -m venv venv && $(ACTIVATE) && pip install -r requirements.txt && python -m experiments.run_experiments

clean:
	rm -rf code/venv
