# References

Place normalized HFSS and openEMS exports here when running benchmark
correlation. The repository does not include proprietary solver project files or
golden results by default.

Expected layout:

```text
references/
└── horn_xband/
    ├── hfss/
    │   ├── sparams.csv
    │   ├── field_near.npz
    │   └── field_far.npz
    └── openems/
        ├── sparams.csv
        ├── field_near.npz
        └── field_far.npz
```

Use `python -m em_solver.benchmark list-cases` to see required artifacts for
each case.
