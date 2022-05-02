(nap0)=

# NAP 0 — Purpose and Process

```{eval-rst}
:Author: "Juan Nunez-Iglesias <mailto:jni@fastmail.com>"
:Author: "Andy Sweet <mailto:andrewdsweet@gmail.com>"
:Author: "Kevin Yamauchi <mailto:kevin.yamauchi@gmail.com>"
:Created: '2022-03-23'
:Status: Active
:Type: Process
```

## What is a NAP?

NAP stands for Napari Advancement Proposal. A NAP is a design document
providing information to the community, or describing a new feature for
napari, its processes, or its environment. (See "Scope of NAPs", below.) The
NAP should provide a rationale for the proposed change as well as a concise
technical specification, if applicable.

We intend NAPs to be the primary mechanisms for proposing major new
features, for collecting community input on an issue, and for documenting
the design decisions that have gone into napari. The NAP author is
responsible for building consensus within the community and documenting
dissenting opinions.

Because the NAPs are maintained as text files in a versioned repository,
their revision history is the historical record of the feature proposal
[^id3].

### Scope of NAPs

The napari project has grown in scope beyond just the software living in the
github.com/napari/napari repository. It includes several other software
packages such as npe2 (napari plugin engine 2), superqt, and magicgui, tools
for the community such as the cookiecutter napari plugin, a webpage, and a chat
forum, among others.

Additionally, napari sits at the center of a much broader community of users,
plugin developers, educators, and downstream or helper libraries.

The scope of NAPs is not strictly defined. Certainly, any controversial
decisions about code changes to the main napari software should be documented
in a NAP. (See "When is a NAP warranted?", below.) Changes to related libraries
in the napari organization may or may not need a NAP, depending on how much the
change would impact the main napari software. Changes to upstream dependencies
outside of the napari organization, such as the napari-hub, fall under the
governance of their respective organizations.

In some cases, the authors of external software or APIs may plan to make a
change that affects the napari community, and may want the feedback of the
napari developers and broader community. They may then choose to create a NAP
as a way to document the plan, solicit feedback, and record the feedback and
any final decisions. In general, the napari developers encourage and appreciate
such engagement with the community as a way to build consensus and drive the
ecosystem forward, together.

### When is a NAP warranted?

Most contributions to napari should go through the standard [contributing
process](napari-contributing), that is, opening a pull request to the main
repository. They will typically be uncontroversial improvements, and
require little design discussion. The git commit and the pull request
itself serve as an adequate record of the contribution's history.

In some cases, contributions will require extensive discussions around any new
APIs, breaking of existing APIs, changing of governance, build, or contributing
mechanisms, or other aspects of the project. These might happen both on the NAP
pull request (PR) itself and on other channels, such as Zulip, community
meetings, or even one on one discussions. In such situations, the PR will not
contain sufficient information to document all the considerations that went
into a decision. Core developers may at their discretion then call for a NAP to
summarize the discussion to date.

In addition to the above situation, napari is the product of many
historical decisions that were *not* explained by a NAP. In some situations,
community members might be confused about parts of napari's design,
and whether alternate designs were considered and rejected, or simply not
considered. It might then be useful to write a retrospective informational
NAP to explain that aspect of the project.

### Types of NAPs

There are three kinds of NAPs:

1. A **Standards Track** NAP describes a new feature or implementation
   for napari.
2. An **Informational** NAP describes a napari design issue, or provides
   general guidelines or information to the napari community, but does not
   propose a new feature. Informational NAPs do not necessarily represent a
   napari community consensus or recommendation, so users and
   implementers are free to ignore Informational NAPs. They may however be used
   to build consensus around conventions or practices. As an example,
   [PEP-257](https://peps.python.org/pep-0257) is an informational PEP
   describing formatting and grammar conventions for docstrings. It specifies
   that the first line of docstrings should be a complete one-line summary of
   the functionality of the function or class. Because the Python standard
   library and many other packages follow this PEP, Jupyter built the
   functionality of pressing Shift-TAB to display just one line of the
   docstring of the item under the cursor.
3. A **Process** NAP describes a process surrounding napari, or
   proposes a change to (or an event in) a process. Process NAPs are
   like Standards Track NAPs but apply to areas other than the napari
   library itself. They may propose an implementation, but not to
   napari's codebase; they require community consensus. Examples include
   procedures, guidelines, changes to the
   {ref}`decision-making process <napari-governance>`, and
   changes to the tools or environment used in napari development.
   Any meta-NAP is also considered a Process NAP.

## NAP Workflow

The NAP process begins with a new idea for napari. A NAP should contain a
single key proposal or new idea. Small enhancements or patches often don't
need a NAP and can be injected into the napari development workflow with a
pull request to the napari [repo]. The more focused the NAP, the more
likely it is to be accepted.

Each NAP must have a champion---someone who writes the NAP using the style
and format described below, shepherds the discussions in the appropriate
forums, and attempts to build community consensus around the idea. The NAP
champion (a.k.a. Author) should first attempt to ascertain whether the idea
is suitable for a NAP. Posting to the napari [issues list] is the best
way to do this.

The proposal should be submitted as a draft NAP via a [GitHub pull
request][github pull request] to the `docs/source/naps` directory with the
name `nap-<n>-<short-title>.md` where `<n>` is an appropriately assigned
number (typically sequential) and `<short-title>` is a one or two word title
for the idea (e.g., `nap-35-lazy-slicing.md`). The draft must use the
{ref}`nap-template` file.

Once the PR is in place, the NAP should be announced on various channels
for discussion, including the
[#naps channel on Zulip](https://napari.zulipchat.com/#narrow/stream/322105-naps) and, if the NAP
has significant user implications, on the
[image.sc forum](https://forum.image.sc/).

At the earliest convenience, the PR should be merged (regardless of whether
it is accepted during discussion). A NAP that outlines a coherent argument
and that is considered reasonably complete should be merged optimistically,
regardless of whether it is accepted during discussion. Additional PRs may
be made by the author to update or expand the NAP, or by maintainers to set
its status, discussion URL, etc.

Standards Track NAPs consist of two parts, a design document and a
reference implementation. It is generally recommended that at least a
prototype implementation be co-developed with the NAP, as ideas that sound
good in principle sometimes turn out to be impractical. Often it makes
sense for the prototype implementation to be made available as a PR to the
napari repo, as long as it is properly marked as WIP (work in progress).

### Review and Resolution

NAPs are discussed in Zulip, on image.sc, and on GitHub. The possible paths
of the status of NAPs are as follows:

```{image} _static/nap-flowchart.png
```

All NAPs should be created with the `Draft` status.

The author of the NAP should periodically update the NAP with new points
both against and in favor of the NAP raised in discussion.

Eventually, after discussion, there may be a consensus that the NAP
should be accepted – see the next section for details. At this point
the status becomes `Accepted`.

Once a NAP has been `Accepted`, the reference implementation must be
completed. When the reference implementation is complete and incorporated
into the main source code repository, the status will be changed to
`Final`.

To allow gathering of additional design and interface feedback before
committing to long term stability for a feature or
API, a NAP may also be marked as "Provisional". This is short for
"Provisionally Accepted", and indicates that the proposal has been accepted
for inclusion in the reference implementation, but additional user feedback
is needed before the full design can be considered "Final". Unlike regular
accepted NAPs, provisionally accepted NAPs may still be Rejected or
Withdrawn even after the related changes have been included in a release.

Wherever possible, it is considered preferable to reduce the scope of a
proposal to avoid the need to rely on the "Provisional" status (e.g. by
deferring some features to later NAPs), as this status can lead to version
compatibility challenges in the wider ecosystem.

A NAP can also be assigned status `Deferred`. The NAP author or a core
developer can assign the NAP this status when no progress is being made on
the NAP.

A NAP can also be `Rejected`. Perhaps after all is said and done it
was not a good idea. It is still important to have a record of this
fact. The `Withdrawn` status is similar---it means that the NAP author
themselves has decided that the NAP is actually a bad idea, or has
accepted that a competing proposal is a better alternative.

When a NAP is `Accepted`, `Deferred`, `Rejected`, or `Withdrawn`, the NAP
should be updated accordingly. In all cases except `Deferred`, the `Resolution`
header should also be added with a link to the relevant post on the discussion
forum. Additional links and information may be added to the Discussion section
of the NAP.

NAPs can also be `Superseded` by a different NAP, rendering the
original obsolete. The `Replaced-By` and `Replaces` headers
should be added to the original and new NAPs respectively.

Process NAPs may also have a status of `Active` if they are never
meant to be completed, e.g. NAP 0 (this NAP).

### How a NAP becomes Accepted

Generally, a NAP is `Accepted` by consensus of all interested contributors. We
therefore need a concrete way to tell whether consensus has been reached. When
you think a NAP is ready to be accepted, start a topic on the Zulip naps
channel with a subject like:

> Proposal to accept NAP #\<number>: \<title>

In the body of the topic, you should:

- link to the latest version of the NAP,
- briefly describe any major points of contention and how they were
  resolved,
- include a sentence like: "If there are no substantive objections
  within 7 days from this post, then the NAP will be accepted; see
  NAP 0 for more details."

For an equivalent example in the NumPy library, see: <https://mail.python.org/pipermail/numpy-discussion/2018-June/078345.html>

After you write the post, you should make sure to link to the specific
thread from the `Discussion` section of the NAP, so that people can
find it later.

Generally the NAP author will be the one to make this post, but
anyone can do it – the important thing is to make sure that everyone
knows when a NAP is on the verge of acceptance, and give them a final
chance to respond. If there's some special reason to extend this final
comment period beyond 7 days, then that's fine, just say so in the
post. You shouldn't do less than 7 days, because sometimes people are
traveling or similar and need some time to respond.

In general, the goal is to make sure that the community has consensus,
not provide a rigid policy for people to try to game. When in doubt,
err on the side of asking for more feedback and looking for
opportunities to compromise.

If the final comment period passes without any substantive objections,
then the NAP can officially be marked `Accepted`. You should send a
follow-up post notifying the thread (celebratory emoji optional but
encouraged 🎉✨), and then update the NAP by setting its `:Status:`
to `Accepted`, and its `:Resolution:` header to a link to your
follow-up post.

If there *are* substantive objections, then the NAP remains in
`Draft` state, discussion continues as normal, and it can be
proposed for acceptance again later once the objections are resolved.

In unusual cases, when no consensus can be reached between core developers,
the [napari Steering Council] may be asked to decide whether a
controversial NAP is accepted, according to our
{ref}`governance <napari-governance>`.

### Maintenance

In general, Standards track NAPs are no longer modified after they have
reached the Final state, as the code and project documentation are
considered the ultimate reference for the implemented feature. They may,
however, be updated under special circumstances.

Process NAPs may be updated over time to reflect changes
to development practices and other details. The precise process followed in
these cases will depend on the nature and purpose of the NAP being updated.

## Format and Template

NAPs are UTF-8 encoded text files using the [MyST markdown] format.  Please
see the {ref}`nap-template` file and the [MyST markdown cheat sheet] for
more information.  We use [Sphinx] to convert NAPs to HTML for viewing on
the web [^id4].

### Header Preamble

Each NAP must begin with a header preamble.  The headers
must appear in the following order.  Headers marked with `*` are
optional.  All other headers are required.

```
  :Author: <list of authors' real names and optionally, email addresses>
  :Status: <Draft | Provisional | Active | Accepted | Deferred | Rejected |
           Withdrawn | Final | Superseded>
  :Type: <Standards Track | Process>
  :Created: <date created on, in yyyy-mm-dd format>
* :Requires: <nap numbers>
* :napari-Version: <version number>
* :Replaces: <nap number>
* :Replaced-By: <nap number>
* :Resolution: <url>
```

The Author header lists the names, and optionally the email addresses of
all the authors of the NAP. The format of the Author header value must be

> Random J. User \<<mailto:address@dom.ain>>

if the email address is included, and just

> Random J. User

if the address is not given.  If there are multiple authors, each should be
on a separate line.

## Discussion

- <https://github.com/napari/napari/pull/4299>

## References and Footnotes

[^id3]: This historical record is available by the normal git commands
    for retrieving older revisions, and can also be browsed on
    [GitHub](https://github.com/napari/napari/tree/main/docs/naps).

[^id4]: The URL for viewing NAPs on the web is
    <https://napari.org/naps/>

## Acknowledgements

This process was based on existing process from the scikit-image (SKIPs), NumPy
(NEPs), and Python (PEPs) projects.

## Copyright

This document has been placed in the public domain.

[developer forum]: https://forum.image.sc/tag/napari
[github pull request]: https://github.com/napari/napari/pulls
[issue tracker]: https://github.com/napari/napari/issues
[repo]: https://github.com/napari/napari
[MyST markdown]: https://myst-parser.readthedocs.io/en/latest/index.html
[MyST markdown cheat sheet]: https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html
[napari steering council]: https://napari.org/community/governance.html
[sphinx]: http://www.sphinx-doc.org/en/stable/
