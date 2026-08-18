# Engine

Internal implementations for BO, LAMMPS evaluation, sampling, surrogate
training, active learning, rescoring and validation. Normal runs should use
`ffopt run ffopt.in --machine PROFILE`; the pipeline invokes these modules and
records their state. Direct module entry points are an internal compatibility
surface for archived campaigns, not the beginner CLI.
