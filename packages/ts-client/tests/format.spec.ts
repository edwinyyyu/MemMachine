import { formatEpisodes, formatSemanticMemories, formatSearchResult } from '@/memory/format'
import type { EpisodicMemory, SemanticMemory, SearchMemoriesResult } from '@/memory/memmachine-memory.types'

describe('formatEpisodes', () => {
  it('returns empty string for empty array', () => {
    expect(formatEpisodes([])).toBe('')
  })

  it('formats a single episode', () => {
    const episode: EpisodicMemory = {
      uid: '1',
      score: 0.9,
      content: 'Hello world',
      created_at: '2024-01-15T13:30:00.000Z',
      producer_id: 'user_1',
      producer_role: 'user',
      episode_type: 'message'
    }
    const result = formatEpisodes([episode])
    expect(result).toBe('[Monday, January 15, 2024 at 01:30 PM] user_1: "Hello world"\n')
  })

  it('formats multiple episodes', () => {
    const episodes: EpisodicMemory[] = [
      {
        uid: '1',
        score: 0.9,
        content: 'First message',
        created_at: '2024-03-05T09:00:00.000Z',
        producer_id: 'user_1',
        producer_role: 'user',
        episode_type: 'message'
      },
      {
        uid: '2',
        score: 0.8,
        content: 'Second message',
        created_at: '2024-03-05T09:01:00.000Z',
        producer_id: 'assistant_1',
        producer_role: 'assistant',
        episode_type: 'message'
      }
    ]
    const result = formatEpisodes(episodes)
    const lines = result.trim().split('\n')
    expect(lines).toHaveLength(2)
    expect(lines[0]).toContain('user_1')
    expect(lines[1]).toContain('assistant_1')
  })

  it('JSON-escapes content', () => {
    const episode: EpisodicMemory = {
      uid: '1',
      score: 0.9,
      content: 'She said "hello"',
      created_at: '2024-01-01T00:00:00.000Z',
      producer_id: 'user_1',
      producer_role: 'user',
      episode_type: 'message'
    }
    const result = formatEpisodes([episode])
    expect(result).toContain(JSON.stringify('She said "hello"'))
  })
})

describe('formatSemanticMemories', () => {
  it('returns empty object for empty array', () => {
    expect(formatSemanticMemories([])).toBe('{}')
  })

  it('formats a single feature', () => {
    const feature: SemanticMemory = {
      set_id: 'set_1',
      category: 'profile',
      tag: 'preferences',
      feature_name: 'favorite_food',
      value: 'pizza',
      metadata: {}
    }
    const result = formatSemanticMemories([feature])
    expect(JSON.parse(result)).toEqual({ preferences: { favorite_food: 'pizza' } })
  })

  it('groups features by tag', () => {
    const features: SemanticMemory[] = [
      {
        set_id: 'set_1',
        category: 'profile',
        tag: 'preferences',
        feature_name: 'food',
        value: 'pizza',
        metadata: {}
      },
      {
        set_id: 'set_1',
        category: 'profile',
        tag: 'preferences',
        feature_name: 'color',
        value: 'blue',
        metadata: {}
      },
      {
        set_id: 'set_1',
        category: 'profile',
        tag: 'background',
        feature_name: 'role',
        value: 'engineer',
        metadata: {}
      }
    ]
    const result = formatSemanticMemories(features)
    expect(JSON.parse(result)).toEqual({
      preferences: { food: 'pizza', color: 'blue' },
      background: { role: 'engineer' }
    })
  })

  it('excludes metadata', () => {
    const feature: SemanticMemory = {
      set_id: 'set_1',
      category: 'profile',
      tag: 'info',
      feature_name: 'name',
      value: 'Alice',
      metadata: {
        id: 'feat_1',
        citations: ['ep_1', 'ep_2'],
        other: { source: 'conversation' }
      }
    }
    const result = formatSemanticMemories([feature])
    expect(JSON.parse(result)).toEqual({ info: { name: 'Alice' } })
    expect(result).not.toContain('set_id')
    expect(result).not.toContain('citations')
    expect(result).not.toContain('feat_1')
  })
})

describe('formatSearchResult', () => {
  it('returns empty string for empty result', () => {
    const result: SearchMemoriesResult = {
      status: 0,
      content: {
        episodic_memory: {
          long_term_memory: { episodes: [] },
          short_term_memory: { episodes: [], episode_summary: [] }
        },
        semantic_memory: []
      }
    }
    expect(formatSearchResult(result)).toBe('')
  })

  it('formats episodic only', () => {
    const result: SearchMemoriesResult = {
      status: 0,
      content: {
        episodic_memory: {
          long_term_memory: {
            episodes: [
              {
                uid: '1',
                score: 0.9,
                content: 'Hello',
                created_at: '2024-01-01T12:00:00.000Z',
                producer_id: 'user_1',
                producer_role: 'user',
                episode_type: 'message'
              }
            ]
          },
          short_term_memory: { episodes: [], episode_summary: [] }
        },
        semantic_memory: []
      }
    }
    const formatted = formatSearchResult(result)
    expect(formatted).toMatch(/^\[Episodic Memory\]\n/)
    expect(formatted).toContain('user_1')
    expect(formatted).toContain('"Hello"')
    expect(formatted).not.toContain('[Semantic Memory]')
  })

  it('formats semantic only', () => {
    const result: SearchMemoriesResult = {
      status: 0,
      content: {
        episodic_memory: {
          long_term_memory: { episodes: [] },
          short_term_memory: { episodes: [], episode_summary: [] }
        },
        semantic_memory: [
          {
            set_id: 'set_1',
            category: 'profile',
            tag: 'prefs',
            feature_name: 'food',
            value: 'pizza',
            metadata: {}
          }
        ]
      }
    }
    const formatted = formatSearchResult(result)
    expect(formatted).toMatch(/^\[Semantic Memory\]\n/)
    expect(formatted).not.toContain('[Episodic Memory]')
    const semanticJson = formatted.replace('[Semantic Memory]\n', '')
    expect(JSON.parse(semanticJson)).toEqual({ prefs: { food: 'pizza' } })
  })

  it('formats combined result', () => {
    const result: SearchMemoriesResult = {
      status: 0,
      content: {
        episodic_memory: {
          long_term_memory: {
            episodes: [
              {
                uid: '1',
                score: 0.9,
                content: 'I like pizza',
                created_at: '2024-01-01T12:00:00.000Z',
                producer_id: 'user_1',
                producer_role: 'user',
                episode_type: 'message'
              }
            ]
          },
          short_term_memory: { episodes: [], episode_summary: [] }
        },
        semantic_memory: [
          {
            set_id: 'set_1',
            category: 'profile',
            tag: 'prefs',
            feature_name: 'food',
            value: 'pizza',
            metadata: {}
          }
        ]
      }
    }
    const formatted = formatSearchResult(result)
    expect(formatted).toContain('[Episodic Memory]')
    expect(formatted).toContain('[Semantic Memory]')
    expect(formatted.indexOf('[Episodic Memory]')).toBeLessThan(formatted.indexOf('[Semantic Memory]'))
  })
})
