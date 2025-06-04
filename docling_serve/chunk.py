import re
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import tiktoken
from docling_serve.datamodel.convert import FileItemChunk


class OverlapStrategy(Enum):
    """重叠策略枚举"""
    FIXED = "fixed"      # 固定token数量重叠
    SEMANTIC = "semantic"  # 语义相关重叠


@dataclass
class ChunkingConfig:
    """分块配置类"""
    max_tokens: int = 512
    overlap_tokens: int = 50
    min_chunk_tokens: int = 20
    prefer_semantic_boundaries: bool = True
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    preserve_lists: bool = True
    overlap_strategy: OverlapStrategy = OverlapStrategy.SEMANTIC
    encoding_name: str = "cl100k_base"


class MarkdownChunker:
    """优化的Markdown文本分块器"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.encoder = tiktoken.get_encoding(self.config.encoding_name)
        # 预编译正则表达式
        self._sentence_pattern = re.compile(r'([。！？.!?])')
        self._header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        self._list_pattern = re.compile(r'^(\s*)([-*+]|\d+\.)\s+(.+)$')
        self._code_block_pattern = re.compile(r'^```')
        self._table_separator_pattern = re.compile(r'^:?-+:?$')
        
        # 缓存token计算结果
        self._token_cache: Dict[str, int] = {}
    
    def count_tokens(self, text: str) -> int:
        """计算文本的tokens数量，带缓存"""
        if text in self._token_cache:
            return self._token_cache[text]
        
        token_count = len(self.encoder.encode(text))
        self._token_cache[text] = token_count
        return token_count
    
    def is_header(self, text: str) -> Tuple[bool, int]:
        """判断是否为标题，返回(是否为标题, 标题级别)"""
        text = text.strip()
        match = self._header_pattern.match(text)
        if match:
            return True, len(match.group(1))
        return False, 0
    
    def is_list_item(self, text: str) -> Tuple[bool, int]:
        """判断是否为列表项，返回(是否为列表项, 缩进级别)"""
        match = self._list_pattern.match(text)
        if match:
            indent_level = len(match.group(1)) // 2  # 假设每级缩进2个空格
            return True, indent_level
        return False, 0
    
    def is_code_block_delimiter(self, text: str) -> bool:
        """判断是否为代码块分隔符"""
        return bool(self._code_block_pattern.match(text.strip()))
    
    def is_table_row(self, text: str) -> bool:
        """
        改进的表格行判断逻辑
        """
        text = text.strip()
        if not text or not text.startswith('|'):
            return False
        
        # 检查是否在代码块内（简单检查反引号数量）
        if text.count('`') % 2 != 0:
            return False
        
        # 分离单元格
        cells = text.split('|')[1:-1]  # 去掉首尾可能的空cell
        if not cells:
            return False
        
        cells = [cell.strip() for cell in cells]
        
        # 检查是否为分隔行
        if all(self._table_separator_pattern.match(cell) for cell in cells):
            return True
        
        # 检查是否为数据行（至少有一个非空单元格）
        return any(cell for cell in cells)
    
    def split_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        sentences = self._sentence_pattern.split(text)
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1])
        return [s for s in result if s.strip()]
    
    def find_semantic_boundary(self, paragraphs: List[str], start_idx: int) -> int:
        """
        寻找语义边界，优先选择标题、列表结束等位置
        """
        if not self.config.prefer_semantic_boundaries:
            return start_idx
        
        # 向前查看几个段落，寻找合适的分割点
        look_ahead = min(3, len(paragraphs) - start_idx)
        
        for i in range(start_idx, start_idx + look_ahead):
            para = paragraphs[i].strip()
            if not para:
                continue
                
            # 优先在标题前分割
            if self.is_header(para)[0]:
                return i
            
            # 在列表结束后分割
            if i > 0 and self.is_list_item(paragraphs[i-1])[0] and not self.is_list_item(para)[0]:
                return i
        
        return start_idx
    
    def collect_complete_structure(self, paragraphs: List[str], start_idx: int) -> Tuple[List[str], int]:
        """
        收集完整的结构（表格、列表、代码块等）
        返回(结构内容, 下一个索引)
        """
        if start_idx >= len(paragraphs):
            return [], start_idx
        
        current_para = paragraphs[start_idx].strip()
        
        # 处理代码块
        if self.config.preserve_code_blocks and self.is_code_block_delimiter(current_para):
            code_block = [current_para]
            i = start_idx + 1
            while i < len(paragraphs):
                code_block.append(paragraphs[i])
                if self.is_code_block_delimiter(paragraphs[i].strip()):
                    break
                i += 1
            return code_block, i + 1
        
        # 处理表格
        if self.config.preserve_tables and self.is_table_row(current_para):
            table_rows = [current_para]
            i = start_idx + 1
            while i < len(paragraphs) and self.is_table_row(paragraphs[i].strip()):
                table_rows.append(paragraphs[i].strip())
                i += 1
            return table_rows, i
        
        # 处理列表
        if self.config.preserve_lists and self.is_list_item(current_para)[0]:
            list_items = [current_para]
            current_level = self.is_list_item(current_para)[1]
            i = start_idx + 1
            
            while i < len(paragraphs):
                para = paragraphs[i].strip()
                if not para:
                    i += 1
                    continue
                    
                is_list, level = self.is_list_item(para)
                if is_list and level >= current_level:
                    list_items.append(para)
                else:
                    break
                i += 1
            return list_items, i
        
        # 普通段落
        return [current_para], start_idx + 1
    
    def get_semantic_overlap(self, current_chunk: List[str], overlap_tokens: int) -> List[str]:
        """
        获取语义相关的重叠内容
        """
        if not current_chunk or self.config.overlap_strategy == OverlapStrategy.FIXED:
            return self.get_fixed_overlap(current_chunk, overlap_tokens)
        
        overlap_paras = []
        total_tokens = 0
        
        # 从后向前寻找合适的重叠内容
        for i in range(len(current_chunk) - 1, -1, -1):
            para = current_chunk[i]
            para_tokens = self.count_tokens(para)
            
            # 如果是标题，优先包含
            if self.is_header(para)[0] and total_tokens + para_tokens <= overlap_tokens:
                overlap_paras.insert(0, para)
                total_tokens += para_tokens
                break  # 找到标题就停止
            
            # 如果添加当前段落不超过限制
            if total_tokens + para_tokens <= overlap_tokens:
                overlap_paras.insert(0, para)
                total_tokens += para_tokens
            else:
                break
        
        return overlap_paras if overlap_paras else self.get_fixed_overlap(current_chunk, overlap_tokens)
    
    def get_fixed_overlap(self, current_chunk: List[str], overlap_tokens: int) -> List[str]:
        """
        获取固定数量的重叠内容（原有逻辑优化版）
        """
        if not current_chunk:
            return []
        
        overlap_paras = []
        total_tokens = 0
        
        # 从后向前收集段落
        for para in reversed(current_chunk):
            para_tokens = self.count_tokens(para)
            
            if not overlap_paras and para_tokens > overlap_tokens:
                # 单个段落过长，按句子截取
                sentences = self.split_sentences(para)
                if not sentences:
                    # 如果无法按句子分割，直接截取tokens
                    tokens = self.encoder.encode(para)
                    overlap_text = self.encoder.decode(tokens[-overlap_tokens:])
                    overlap_paras.append(overlap_text.lstrip())
                else:
                    # 按句子收集
                    selected_sentences = []
                    current_tokens = 0
                    
                    for sent in reversed(sentences):
                        sent_tokens = self.count_tokens(sent)
                        if current_tokens + sent_tokens <= overlap_tokens:
                            selected_sentences.insert(0, sent)
                            current_tokens += sent_tokens
                        else:
                            break
                    
                    if selected_sentences:
                        overlap_paras.append(''.join(selected_sentences).lstrip())
                break
            
            elif total_tokens + para_tokens <= overlap_tokens:
                overlap_paras.insert(0, para)
                total_tokens += para_tokens
            else:
                break
        
        return overlap_paras
    
    def evaluate_chunk_quality(self, chunk_content: str) -> bool:
        """
        评估块的质量
        """
        if not chunk_content.strip():
            return False
        
        lines = chunk_content.strip().split('\n')
        token_count = self.count_tokens(chunk_content)
        
        # 检查最小token数量
        if token_count < self.config.min_chunk_tokens:
            return False
        
        # 避免只有标题的块
        if len(lines) == 1 and self.is_header(lines[0])[0]:
            return False
        
        return True
    
    def split_text(self, content_md: str) -> List[FileItemChunk]:
        """
        主要的文本分块函数
        """
        if not content_md.strip():
            return []
        
        # 清理和预处理
        paragraphs = [p for p in re.split(r'\n\n|\n', content_md.strip()) if p.strip()]
        
        chunks_list = []
        current_chunk = []
        current_tokens = 0
        i = 0
        
        while i < len(paragraphs):
            # 收集完整结构
            structure_content, next_i = self.collect_complete_structure(paragraphs, i)
            
            if not structure_content:
                i = next_i
                continue
            
            structure_text = '\n'.join(structure_content)
            structure_tokens = self.count_tokens(structure_text)
            
            # 检查是否需要分块
            if current_tokens + structure_tokens > self.config.max_tokens and current_chunk:
                # 寻找语义边界
                boundary_idx = self.find_semantic_boundary(paragraphs, i)
                
                # 保存当前块
                chunk_content = '\n'.join(current_chunk)
                if self.evaluate_chunk_quality(chunk_content):
                    chunks_list.append(FileItemChunk(
                        content=chunk_content,
                        tokens=current_tokens
                    ))
                
                # 获取重叠内容
                if self.config.overlap_strategy == OverlapStrategy.SEMANTIC:
                    overlap_content = self.get_semantic_overlap(current_chunk, self.config.overlap_tokens)
                else:
                    overlap_content = self.get_fixed_overlap(current_chunk, self.config.overlap_tokens)
                
                # 开始新块
                current_chunk = overlap_content.copy()
                current_tokens = self.count_tokens('\n'.join(current_chunk)) if current_chunk else 0
            
            # 添加当前结构到块中
            if structure_tokens > self.config.max_tokens:
                # 结构本身就很大，需要特殊处理
                if current_chunk:
                    # 先保存当前块
                    chunk_content = '\n'.join(current_chunk)
                    if self.evaluate_chunk_quality(chunk_content):
                        chunks_list.append(FileItemChunk(
                            content=chunk_content,
                            tokens=current_tokens
                        ))
                    current_chunk = []
                    current_tokens = 0
                
                # 将大结构作为单独的块
                chunks_list.append(FileItemChunk(
                    content=structure_text,
                    tokens=structure_tokens
                ))
            else:
                current_chunk.extend(structure_content)
                current_tokens += structure_tokens
            
            i = next_i
        
        # 处理最后一个块
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            if self.evaluate_chunk_quality(chunk_content):
                chunks_list.append(FileItemChunk(
                    content=chunk_content,
                    tokens=current_tokens
                ))
        
        return chunks_list
    
    def get_chunk_statistics(self, chunks: List[FileItemChunk]) -> Dict:
        """获取分块统计信息"""
        if not chunks:
            return {}
        
        token_counts = [chunk.tokens for chunk in chunks]
        return {
            'total_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'avg_tokens_per_chunk': sum(token_counts) / len(chunks),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'cache_hit_rate': len(self._token_cache) / max(1, len(token_counts))
        }


def create_chunker(max_tokens: int = 512, 
                  overlap_tokens: int = 50,
                  overlap_strategy: str = "semantic",
                  preserve_structure: bool = True) -> MarkdownChunker:
    """
    创建分块器的便利函数
    """
    config = ChunkingConfig(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        overlap_strategy=OverlapStrategy.SEMANTIC if overlap_strategy == "semantic" else OverlapStrategy.FIXED,
        prefer_semantic_boundaries=preserve_structure,
        preserve_code_blocks=preserve_structure,
        preserve_tables=preserve_structure,
        preserve_lists=preserve_structure
    )
    return MarkdownChunker(config)


def process_markdown(content_md: str, 
                    max_tokens: int = 512, 
                    overlap_tokens: int = 50,
                    overlap_strategy: str = "semantic",
                    show_stats: bool = True) -> List[FileItemChunk]:
    """
    处理markdown文本的便利函数
    """
    chunker = create_chunker(max_tokens, overlap_tokens, overlap_strategy)
    chunks = chunker.split_text(content_md)
    
    if show_stats:
        stats = chunker.get_chunk_statistics(chunks)
        print(f"分块统计信息:")
        print(f"- 总块数: {stats.get('total_chunks', 0)}")
        print(f"- 总tokens: {stats.get('total_tokens', 0)}")
        print(f"- 平均每块tokens: {stats.get('avg_tokens_per_chunk', 0):.1f}")
        print(f"- tokens范围: {stats.get('min_tokens', 0)} - {stats.get('max_tokens', 0)}")
        print(f"- 缓存命中率: {stats.get('cache_hit_rate', 0):.2%}")
        print()
        
        for i, chunk in enumerate(chunks, 1):
            print(f"=== 块 {i} (tokens: {chunk.tokens}) ===")
            print(chunk.content)
            print()
    
    return chunks


# 使用示例
if __name__ == "__main__":
    # 示例1: 基本使用
    sample_md = """
# 标题1

这是第一个段落。

## 标题2

这是第二个段落，包含一些内容。

| 列1 | 列2 | 列3 |
|-----|-----|-----|
| 数据1 | 数据2 | 数据3 |
| 数据4 | 数据5 | 数据6 |

- 列表项1
- 列表项2
  - 子列表项1
  - 子列表项2

```python
def example():
    print("代码块示例")
    return True
```

最后一个段落。
"""
    
    print("=== 基本使用示例 ===")
    chunks = process_markdown(sample_md, max_tokens=200, overlap_tokens=30)
    
    # 示例2: 自定义配置
    print("\n=== 自定义配置示例 ===")
    config = ChunkingConfig(
        max_tokens=300,
        overlap_tokens=50,
        overlap_strategy=OverlapStrategy.SEMANTIC,
        prefer_semantic_boundaries=True,
        min_chunk_tokens=50
    )
    
    chunker = MarkdownChunker(config)
    chunks2 = chunker.split_text(sample_md)
    stats = chunker.get_chunk_statistics(chunks2)
    print(f"自定义配置结果: {stats}")