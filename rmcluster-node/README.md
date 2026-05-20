# rmcluster-node Theos Scaffold

This directory is the NIC-generated scaffold for the legacy app.

The real build configuration for the ported app lives at the repo root in:

- [`../Makefile`](../Makefile)
- [`../control`](../control)

The local [`Makefile`](./Makefile) is only a convenience wrapper that forwards
common Theos targets back to the repo root, so both of these work:

```sh
make -C .. ipa
make ipa
```

from inside `rmcluster-node/`.
