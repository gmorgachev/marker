from marker.schema import BlockTypes
from marker.schema.blocks.basetable import BaseTable


class TableOfContents(BaseTable):
    block_type: str = BlockTypes.TableOfContents
