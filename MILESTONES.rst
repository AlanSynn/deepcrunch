==========
Milestones
==========

.. list-table::

  * - Version
    - Status
    - Author
    - Date
  * - 0.1.0
    - Draft
    - `Alan Synn`_
    - 2023-07-20

.. _Alan Synn: mailto:alan@alansynn.com

Logistics
=========

**VCS**

1. Github
    - Create as private and send invitations
    - Necessary information for basic usage

**Documentation**

1. Reports should be in Confluence format
2. Build manuals in a `docs` fashion

**Testing**

1. Focus mainly on open models
    - e.g. Vicuna - 5B, ResNet-152
    - Discuss when the package is out
    - Revisit in the Week4
2. The format should be similar to previous Confluence materials
    - `Confluence Reference`_

.. _Confluence Reference: https://confluence.dx.lguplus.co.kr/pages/viewpage.action?pageId=179265360

Outcome
=======

- Midterm presentation in the third week
    - Presentation of progress
        - Design (DX): 50%
        - Overall library structure: 50%
        - Plan
        - Technical aspects (anticipated performance or midterm presentation)

- Brief briefing on trends or research
    - More focused on trending things
        - Retraining
        - Continual learning

- Design goal and objectives
    - Expandability
        - Compression algorithms (in-training/inference)
        - Scheduler (in training)
        - Accelerator (CPU/GPU)
        - If there is more time: multi-gpu serving
        - `PyTorch Pipeline Reference`_

.. _PyTorch Pipeline Reference: https://pytorch.org/docs/stable/pipeline.html#pipeline-parallelism
