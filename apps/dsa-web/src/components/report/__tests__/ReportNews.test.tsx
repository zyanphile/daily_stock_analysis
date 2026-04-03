import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { historyApi } from '../../../api/history';
import { ReportNews } from '../ReportNews';

vi.mock('../../../api/history', () => ({
  historyApi: {
    getNews: vi.fn(),
  },
}));

describe('ReportNews', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders news items and refreshes with preserved subpanel styling', async () => {
    vi.mocked(historyApi.getNews).mockResolvedValue({
      total: 1,
      items: [
        {
          title: '茅台发布最新经营数据',
          snippet: '公司披露季度经营情况，市场关注度提升。',
          url: 'https://example.com/news',
        },
      ],
    });

    const { container } = render(<ReportNews recordId={1} />);

    expect(await screen.findByText('茅台发布最新经营数据')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '跳转' })).toHaveAttribute('href', 'https://example.com/news');
    expect(container.querySelector('.home-panel-card')).toBeTruthy();
    expect(container.querySelector('.home-subpanel')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: '刷新' }));

    await waitFor(() => {
      expect(historyApi.getNews).toHaveBeenCalledTimes(2);
    });
  });

  it('renders the empty state when no news exists', async () => {
    vi.mocked(historyApi.getNews).mockResolvedValue({
      total: 0,
      items: [],
    });

    render(<ReportNews recordId={1} />);

    expect(await screen.findByText('暂无相关资讯')).toBeInTheDocument();
    expect(screen.getByText('可稍后刷新以获取最新资讯。')).toBeInTheDocument();
  });

  it('uses history-specific empty copy for saved report snapshots', async () => {
    vi.mocked(historyApi.getNews).mockResolvedValue({
      total: 0,
      items: [],
    });

    render(<ReportNews recordId={1} isHistory />);

    expect(await screen.findByText('暂无相关资讯')).toBeInTheDocument();
    expect(screen.getByText('该历史记录未保存相关资讯；重新读取只会刷新当前快照。若要获取最新资讯，请重新发起一次分析。')).toBeInTheDocument();
  });

  it('clarifies that history reports only reload the saved news snapshot', async () => {
    vi.mocked(historyApi.getNews).mockResolvedValue({
      total: 1,
      items: [
        {
          title: '历史快照中的资讯',
          snippet: '这条资讯来自已保存的历史记录。',
          url: 'https://example.com/history-news',
        },
      ],
    });

    render(<ReportNews recordId={1} isHistory />);

    expect(await screen.findByText('历史快照中的资讯')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('历史资讯快照');
    expect(screen.getByRole('alert')).toHaveTextContent('这里展示的是分析生成时保存的资讯结果');

    fireEvent.click(screen.getByRole('button', { name: '重新读取' }));

    await waitFor(() => {
      expect(historyApi.getNews).toHaveBeenCalledTimes(2);
    });
  });

  it('localizes the empty state description for english reports', async () => {
    vi.mocked(historyApi.getNews).mockResolvedValue({
      total: 0,
      items: [],
    });

    render(<ReportNews recordId={1} language="en" />);

    expect(await screen.findByText('No related news')).toBeInTheDocument();
    expect(screen.getByText('Refresh later to check for the latest updates.')).toBeInTheDocument();
  });

  it('renders the error state and supports retry', async () => {
    vi.mocked(historyApi.getNews)
      .mockRejectedValueOnce(new Error('network failed'))
      .mockResolvedValueOnce({
        total: 1,
        items: [
          {
            title: '重试成功',
            snippet: '第二次请求成功返回。',
            url: 'https://example.com/retry',
          },
        ],
      });

    render(<ReportNews recordId={1} />);

    expect(await screen.findByRole('alert')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '重试' }));

    expect(await screen.findByText('重试成功')).toBeInTheDocument();
  });
});
