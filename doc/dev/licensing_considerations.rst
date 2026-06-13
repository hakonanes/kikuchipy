Licensing considerations
========================

As mentioned :doc:`in the user guide </user/licensing>`, kikuchipy as a whole carries
the GPLv3+ license, while select components carry a more permissive BSD 3-Clause
license.

For developers, this means:

- When introducing new functionality to kikuchipy, we have the option of giving it the
  more permissive BSD license.
- To honor the more permissive license, we cannot import from GPL licensed code in the
  files and directories that carry the BSD license.

For reviewers, this means we must consider the following when reviewing a PR:

- If used, does the code honor the more permissive BSD license?
- Does the contributor know about the option of licensing using BSD instead of GPL?
  If not, we can suggest this.

Why this change from all GPL to GPL + some BSD?
Some arguments are given in a comment on `this RosettaSciIO discussion about licensing
<https://github.com/hyperspy/rosettasciio/issues/51#issuecomment-4634847410>`.
If you have thoughts or concerns regarding licensing, feel free to `open an issue on
our GitHub repo <https://github.com/pyxem/kikuchipy/issues/new>`_ or send an email to
the maintainers at pyxem.team@gmail.com.
