PYTHON ?= python3
RUN = PYTHONPATH=src $(PYTHON) -m

.PHONY: all data pipeline dq models validate figures report test clean

all: data pipeline dq models validate figures report

data:        ## download the UCI file once and verify checksums
	$(RUN) cdmv.download

pipeline:    ## load raw table, run SQL staging and features
	$(RUN) cdmv.pipeline

dq:          ## SQL data-quality checks; exits non-zero on any error-level failure
	$(RUN) cdmv.dq

models: dq   ## fit champion and challenger, score every split
	$(RUN) cdmv.train

validate: models   ## performance, calibration, stability, sensitivity tables
	$(RUN) cdmv.validate

figures:     ## reports/figures/*.png
	$(RUN) cdmv.figures

report:      ## reports/model_validation.md and README results block
	$(RUN) cdmv.report

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

clean:       ## remove generated warehouse and model files (keeps data/raw)
	rm -rf warehouse output
